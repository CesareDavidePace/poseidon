import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
sys.path.insert(0, os.path.abspath('/home/pace/Poseidon/'))

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.nn.functional as F
from mmpose.apis import init_model
from engine.defaults import default_parse_args
from torchvision.models._utils import IntermediateLayerGetter
import math
from posetimation import get_cfg, update_config 

import torch
import torch.nn as nn
import torch.nn.functional as F

from deformable_attention_2d import DeformableAttention2D

from torchvision.ops import DeformConv2d

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFrameWeighting(nn.Module):
    def __init__(self, embed_dim, num_frames):
        super(AdaptiveFrameWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        self.frame_quality_estimator = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, embed_dim, height, width]
        batch_size, num_frames, embed_dim, height, width = x.shape
        
        # Estimate quality for each frame
        x_reshaped = x.view(batch_size * num_frames, embed_dim, height, width)
        quality_scores = self.frame_quality_estimator(x_reshaped).view(batch_size, num_frames)
        
        # Normalize scores
        weights = F.softmax(quality_scores, dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Weight frames
        weighted_x = x * weights
        
        return weighted_x, weights.squeeze()


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, query, context):
        # Reshape input: [B, C, H, W] -> [H*W, B, C]
        B, num_context_frames, C, H, W = context.shape
        query = query.view(B, C, -1).permute(2, 0, 1) # [H*W, B, C]
        context = context.view(B, num_context_frames, C, -1).permute(3, 0, 1, 2) # [H*W, B, num_context_frames, C]
        context = context.reshape(-1, B, C) # [H*W*num_context_frames, B, C]

        # Apply weighted attention
        attn_output, _ = self.mha(query, context, context,)
        #attn_output = self.dropout(attn_output)

        # Reshape output: [H*W, B, C] -> [B, C, H, W]
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4):
        super(Poseidon, self).__init__()
        
        self.device = device

        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE
   
        # Scongelamento parziale del backbone
        if cfg.MODEL.FREEZE_WEIGHTS:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for layer in self.backbone.layers[-12:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln1.parameters():
                param.requires_grad = True
                

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)
        self.embed_dim = cfg.MODEL.EMBED_DIM # 384
        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS # 17
        self.num_frames = cfg.WINDOWS_SIZE
        
        self.is_train = True if phase == 'train' else False

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Cross-Attention
        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 24, 18])

        # Print learning parameters
        print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon backbone parameters: {round(self.count_backbone_parameters() / 1e6, 1)} M\n\n")

    def count_backbone_parameters(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        x = self.backbone(x)[0]

        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18)

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

        #x = self.dropout(x)
        
        # Cross-Attention
        center_frame_idx = num_frames // 2
        center_frame = x[:, center_frame_idx]
        context_frames = torch.cat([x[:, :center_frame_idx], x[:, center_frame_idx+1:]], dim=1)

        context_frames = context_frames.view(-1, self.embed_dim, 24*18).permute(2, 0, 1)
        context_frames, _ = self.self_attention(context_frames, context_frames, context_frames)
        context_frames = context_frames.permute(1, 2, 0).view(batch_size, num_frames-1, self.embed_dim, 24, 18)

        # Cross-Attention
        attended_features = self.cross_attention(center_frame, context_frames)
        
        # Layer Norm
        attended_features = self.layer_norm(attended_features)

        # residual connection
        attended_features += center_frame

        # Deconvolution layers
        x = self.deconv_layer(attended_features)

        # Final layer
        x = self.final_layer(x)
        
        return x


def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)

    return cfg

def find_max_batch_size(model, max_batch_size=32):
    device = next(model.parameters()).device
    left, right = 1, max_batch_size
    
    while left <= right:
        mid = (left + right) // 2
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(mid, 5, 3, 384, 288).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            left = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                right = mid - 1
            else:
                raise e
    
    max_batch = right
    print(f"Maximum batch size: {max_batch}")
    
    # Now test with the maximum batch size to get the memory usage
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    dummy_input = torch.randn(max_batch, 5, 3, 384, 288).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    print(f"Peak GPU memory usage with batch size {max_batch}: {peak_memory:.2f} MB")

    return max_batch

def test_model():
    
    args = default_parse_args()
    cfg = setup(args)

    #config_path = '/home/pace/Poseidon/configs/configDCPose.yaml'
    #cfg = load_config(config_path)
    phase = 'train'
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Create the model
    model = Poseidon(cfg=cfg).to(device)
    
    dummy_input = torch.randn(1, 5, 3, 384, 288).to(device)

    # Run a forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Calculate the peak memory usage
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    print(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    # gpu memory disponibile
    print(f"Free Memory: {torch.cuda.get_device_properties(1).total_memory / 1024**2} MB")

    # Find the maximum batch size
    max_batch_size = find_max_batch_size(model, max_batch_size=128)


if __name__ == "__main__":
    test_model()


# python models/test_model5.py --config /home/pace/Poseidon/configs/configPoseidon.yaml
