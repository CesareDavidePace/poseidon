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


from posetimation import get_cfg, update_config 

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)


    def forward(self, query, context):
        # Reshape input: [B, C, H, W] -> [H*W, B, C]
        B, num_context_frames, C, H, W = context.shape
        query = query.view(B, C, -1).permute(2, 0, 1) # [H*W, B, C]
        context = context.view(B, num_context_frames, C, -1).permute(3, 0, 1, 2) # [H*W, B, num_context_frames, C]
        context = context.reshape(-1, B, C) # [H*W*num_context_frames, B, C]

        # Apply weighted attention
        attn_output, _ = self.mha(query, context, context,)

        # Reshape output: [H*W, B, C] -> [B, C, H, W]
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output

        

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, embed_dim=384, num_layers=3):
        super(Poseidon, self).__init__()
        
        self.device = device
        config_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
        checkpoint_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'


        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        # Scongelamento parziale del backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.ln1.parameters():
            param.requires_grad = True

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = embed_dim # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_frames = cfg.WINDOWS_SIZE
        self.num_layers = num_layers
        
        self.is_train = True if phase == 'train' else False

        # N layers of Cross-Attention and Self-Attention
        self.cross_attention_layers = nn.ModuleList([CrossAttention(self.embed_dim, self.num_heads) for _ in range(num_layers)])
        self.self_attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads) for _ in range(num_layers)])

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm([embed_dim, 24, 18]) for _ in range(num_layers)])
        
        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone
        x = self.backbone(x)[0]

        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18) # [batch_size, num_frames, 384, 24, 18]

        # Center frame index
        center_frame_idx = num_frames // 2

        # Cross-Attention and Self-Attention layers
        for i in range(self.num_layers):
            center_frame = x[:, center_frame_idx]
            context_frames = torch.cat([x[:, :center_frame_idx], x[:, center_frame_idx+1:]], dim=1)

            # Self-Attention on context frames
            context_frames = context_frames.view(-1, self.embed_dim, 24*18).permute(2, 0, 1)
            context_frames, _ = self.self_attention_layers[i](context_frames, context_frames, context_frames)
            context_frames = context_frames.permute(1, 2, 0).view(batch_size, num_frames-1, self.embed_dim, 24, 18)

            # Cross-Attention
            attended_features = self.cross_attention_layers[i](center_frame, context_frames)
            
            # Layer Norm and residual connection
            attended_features = self.layer_norms[i](attended_features)
            attended_features += center_frame

            # Update x for the next layer
            x[:, center_frame_idx] = attended_features

        # Deconvolution layers
        x = self.deconv_layer(attended_features)

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



