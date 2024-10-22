import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from .backbones import Backbones
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.nn.functional as F
from mmpose.apis import init_model
import math


from posetimation import get_cfg, update_config 


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, H, W]
        batch_size, num_frames, embed_dim, H, W = x.shape
        x = x.view(batch_size, num_frames, embed_dim, -1)  # Flatten spatial dimensions
        x = x.permute(0, 3, 2, 1)  # [batch_size, H*W, embed_dim, num_frames]
        x = x.reshape(batch_size * H * W, embed_dim, num_frames)  # Merge batch and spatial dimensions
        x = x.permute(2, 0, 1)  # [num_frames, batch_size * H * W, embed_dim]

        x = self.transformer_encoder(x)  # [num_frames, batch_size * H * W, embed_dim]

        x = x.permute(1, 2, 0)  # [batch_size * H * W, embed_dim, num_frames]
        x = x.view(batch_size, H * W, embed_dim, num_frames)
        x = x.permute(0, 3, 2, 1)  # [batch_size, num_frames, embed_dim, H*W]
        x = x.view(batch_size, num_frames, embed_dim, H, W)
        return x

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=8, embed_dim=384, num_layers=1):
        super(Poseidon, self).__init__()

        self.device = device
        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        
        self.backbone = self.model.backbone

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Model parameters
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_frames = cfg.WINDOWS_SIZE
        self.is_train = True if phase == 'train' else False


        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(self.embed_dim, self.num_heads, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 16, 12])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")
        print(f"Poseidon total parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone feature extraction
        model_output = self.backbone(x)[0]

        # Reshape model output
        x = model_output.view(batch_size, num_frames, self.embed_dim, 16, 12)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Layer Norm
        x = self.layer_norm(x[:, num_frames // 2])

        # Deconvolution layers
        x = self.deconv_layer(x)

        # Final layer
        x = self.final_layer(x)

        return x

    def set_phase(self, phase):
        self.phase = phase
        self.is_train = True if phase == TRAIN_PHASE else False

    def get_phase(self):
        return self.phase

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            norm_factor=np.ones((N, 2), dtype=np.float32))

        return avg_acc



