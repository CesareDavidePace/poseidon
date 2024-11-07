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
from torchvision.models._utils import IntermediateLayerGetter

class ExtractIntermediateLayers(nn.Module):
    def __init__(self, model, return_layers):
        super(ExtractIntermediateLayers, self).__init__()
        self.model = model
        self.return_layers = return_layers

        self.getter = IntermediateLayerGetter(
            self.model,
            return_layers=self.return_layers
        )

    def forward(self, x):
        outputs = self.getter(x)
        return outputs

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=3, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
        self.device = device
        config_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
        checkpoint_file = '/home/pace/Poseidon/models/vitpose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.backbone = self.model.backbone

        self.return_layers = {'layers.3': 'layer3', 'layers.7': 'layer7', 'layers.11': 'layer11'}

        self.extract_layers = ExtractIntermediateLayers(self.backbone, self.return_layers)

        # print(self.backbone) # torch.Size([batch_size*num_frames, 384, 24, 18])
        
        # Scongelamento parziale del backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for layer in self.backbone.layers[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.ln1.parameters():
            param.requires_grad = True

            
        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.output_sizes = 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        self.embed_dim = self.num_joints * self.embed_dim_for_joint
        
        # Ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # cross-attention layer for frames
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.output_sizes, num_heads=self.num_heads, batch_first=True)

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()
        
        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)
        
        self.is_train = True if phase == 'train' else False
        
        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters() / 1e6, 1)} M\n\n")


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        backbone_outputs = self.backbone(x)[0]  # shape: [batch_size*num_frames, 384, 24, 18]

        # print("Backbone outputs:", backbone_outputs.shape) # torch.Size([batch_size*num_frames, 384, 24, 18])

        # print shape of each feature
        for feature in self.return_layers.values():
            print(f"{feature} shape:", backbone_outputs[feature].shape)

        combined_features = sum([backbone_outputs[feature] for feature in self.return_layers.values()])

        print("Combined features:", combined_features.shape) # torch.Size([batch_size*num_frames, 384, 24, 18])

        # Apply adaptive pooling
        backbone_outputs = self.adaptive_pool(backbone_outputs)  # shape: [batch_size*num_frames, 384, 1, 1]

        x = backbone_outputs.view(batch_size, num_frames, -1)  # shape: [batch_size, num_frames, 384]

        central_frame = x[:, num_frames // 2, :].unsqueeze(1)  # shape: [batch_size, 1, 384]

        # print("Central frame:", central_frame.shape) # torch.Size([batch_size, 1, 384])

        context_frames = x # shape: [batch_size, num_frames, 384]

        # print("Context frames:", context_frames.shape) # torch.Size([batch_size, num_frames, 384])

        # Apply cross-attention
        x, _ = self.cross_attention(central_frame, context_frames, context_frames)  # shape: [batch_size, 1, 384]

        # print("After cross-attention:", x.shape) # torch.Size([batch_size, 1, 384])

        # Apply deconv layers
        x = x.view(batch_size, -1, 1, 1)  # shape: [batch_size, 384, 1, 1]
        x = self.deconv_layers(x)  # shape: [batch_size, 17, 96, 72]

        # print("After deconv layers:", x.shape) # torch.Size([batch_size, 17, 96, 72])

        heatmap = self.final_layer(x)  # shape: [batch_size, 17, 96, 72]

        return heatmap

    def _make_deconv_layers(self):
        layers = []
        input_channels = self.output_sizes  # Adjusted for the flattened input
        upsample_configs = [
            (256, 2),  # [1, 1] -> [2, 2]
            (128, 2),  # [2, 2] -> [4, 4]
            (64, 2),   # [4, 4] -> [8, 8]
            (32, 2),   # [8, 8] -> [16, 16]
            (32, 2),   # [16, 16] -> [32, 32]
            (self.num_joints, 3)  # [32, 32] -> [96, 96]
        ]
        
        for out_channels, scale_factor in upsample_configs:
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, bias=False))
            if out_channels != self.num_joints:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((96, 72)))
        return nn.Sequential(*layers)

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


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6