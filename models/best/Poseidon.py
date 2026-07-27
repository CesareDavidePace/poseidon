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


from posetimation import get_cfg, update_config 

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

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.paths = nn.ModuleList()
        for pool_size in pool_sizes:
            self.paths.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        for path in self.paths:
            features.append(F.interpolate(path(x), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(features, dim=1)

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiScaleFeatureFusion, self).__init__()
        self.ppm = PyramidPoolingModule(embed_dim, embed_dim // 4)
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False)
        self.fusion_norm = nn.BatchNorm2d(embed_dim)
        self.fusion_act = nn.ReLU(inplace=True)
        self.attention_fusion = AttentionFusion(embed_dim, num_heads)


    def forward(self, features):
        # Assume features is a dict of tensors from different layers
        multi_scale_features = {}
        for name, feature in features.items():
            B, num_frames, C, H, W = feature.shape
            feature = feature.view(-1, C, H, W)
            multi_scale_feature = self.ppm(feature)
            multi_scale_feature = self.fusion_conv(multi_scale_feature)
            multi_scale_feature = self.fusion_norm(multi_scale_feature)
            multi_scale_feature = self.fusion_act(multi_scale_feature)
            multi_scale_feature = multi_scale_feature.view(B, num_frames, C, H, W)
            multi_scale_features[name] = multi_scale_feature
        
        # Use the existing AttentionFusion module
        fused_features = self.attention_fusion(multi_scale_features)
        return fused_features

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, features):
        # Assuming features is a dictionary of tensors, each with shape [10, 432, 384]
        B, num_frames, embed_dim, H, W = list(features.values())[0].shape

        # for each item in feature dict
        for key, value in features.items():
            features[key] = value.reshape(B*num_frames, H*W, embed_dim)

        # Concatenate features along the sequence dimension
        features_cat = torch.cat(list(features.values()), dim=1)  # Shape: [10, 432*num_features, 384]

        # get shape of features
        B_numframes, _, embed_dim = features_cat.shape
        
        # Transpose for attention: [432*num_features, 10, 384]
        features_cat = features_cat.transpose(0, 1)
        
        # Apply self-attention
        attn_output, _ = self.attention(features_cat, features_cat, features_cat)
        
        # Add residual connection and layer norm
        fused_features = self.norm(features_cat + attn_output)
        
        # Project back to original sequence length
        fused_features = fused_features.transpose(0, 1)  # Shape: [10, 432*num_features, 384]
        fused_features = fused_features.view(B_numframes, len(features), -1, embed_dim)  # Shape: [10, num_features, 432, 384]
        fused_features = torch.mean(fused_features, dim=1)  # Shape: [10, 432, 384]
        
        # Final projection to ensure we capture information from all feature levels
        fused_features = self.final_proj(fused_features)
        
        return fused_features


class ExtractIntermediateLayers(nn.Module):
    def __init__(self, model, return_layers):
        super(ExtractIntermediateLayers, self).__init__()
        self.model = model
        self.return_layers = return_layers
        self.features = {}

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, layer in self.return_layers.items():
            layer_idx = int(name.split('.')[1])
            self.model.layers[layer_idx].register_forward_hook(get_hook(layer))

    def forward(self, x):
        self.features = {}
        model_output = self.model(x)[0]
        return self.features, model_output

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
        #attn_output = self.dropout(attn_output)

        # Reshape output: [H*W, B, C] -> [B, C, H, W]
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output

class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase='train', num_heads=4, inference=False):
        super(Poseidon, self).__init__()
        
        self.device = device
        self.inference = inference

        self.model = init_model(cfg.MODEL.CONFIG_FILE, cfg.MODEL.CHECKPOINT_FILE, device=device)
        self.backbone = self.model.backbone

        if cfg.MODEL.EMBED_DIM == 384:
            self.return_layers = {'layers.3': 'layer3', 'layers.7': 'layer7'} # VIT S
        elif cfg.MODEL.EMBED_DIM == 768:
            self.return_layers = {'layers.3': 'layer3', 'layers.7': 'layer7'} # VIT B (12 layers)
        elif cfg.MODEL.EMBED_DIM == 1024:
            self.return_layers = {'layers.9': 'layer9', 'layers.21': 'layer21'} # VIT L (24 layers)
        elif cfg.MODEL.EMBED_DIM == 1280:
            self.return_layers = {'layers.9': 'layer9', 'layers.21': 'layer21'} # VIT H (32 layers)

        #self.return_layers = {'layers.7': 'layer7','layers.15':'layers15', 'layers.23': 'layers23'}
        #self.return_layers = {'layers.9': 'layer9', 'layer.21': 'layer21',} # VIT H
        #self.return_layers = {'layers.3': 'layer3', 'layers.7': 'layer7'} # VIT S, VIT B

        self.extract_layers = ExtractIntermediateLayers(self.backbone, self.return_layers)

        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer
        self.num_frames = cfg.WINDOWS_SIZE

        if not self.inference:
            # Scongelamento parziale del backbone
            # check if there is a freeze_weights parameter in the config
            if hasattr(cfg.MODEL, 'FREEZE_WEIGHTS') and cfg.MODEL.FREEZE_WEIGHTS:
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

        # Feature Fusion
        self.feature_fusion = MultiScaleFeatureFusion(self.embed_dim, num_heads=self.num_heads)

        # Adaptive Frame Weighting
        self.adaptive_weighting = AdaptiveFrameWeighting(self.embed_dim, self.num_frames)

        # Cross-Attention
        self.cross_attention = CrossAttention(self.embed_dim, self.num_heads)

        # Self-Attention
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        #self.dropout = nn.Dropout(0.1)

        # Layer norms for intermediate features
        self.intermediate_layer_norms = nn.ModuleDict({
            name: self.backbone.ln1 if name == 'layer31' else nn.LayerNorm(self.embed_dim)
            for name in self.return_layers.values()
        })

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.embed_dim, 24, 18])

        # load pretrained weights if available cfg.MODEL.PRETRAINED
        if hasattr(cfg.MODEL, 'PRETRAINED') and cfg.MODEL.PRETRAINED:
            pretrained_path = cfg.MODEL.PRETRAINED
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                self.load_state_dict(torch.load(pretrained_path, map_location=self.device), strict=False)
            else:
                print(f"Pretrained weights not found at {pretrained_path}, using random initialization.")



        # Print learning parameters
        if not self.inference:
            print(f"Poseidon learnable parameters: {round(self.count_trainable_parameters() / 1e6, 1)} M\n\n")

        print(f"Poseidon parameters: {round(self.count_parameters() / 1e6, 1)} M\n\n")

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    def forward(self, x, meta=None, return_weights=False):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        # Backbone
        intermediate_outputs, model_output = self.extract_layers(x)

        # Apply LayerNorm to the extracted features
        for feature_name in intermediate_outputs.keys():
            intermediate_outputs[feature_name] = self.intermediate_layer_norms[feature_name](intermediate_outputs[feature_name])
            intermediate_outputs[feature_name] = intermediate_outputs[feature_name].view(batch_size, num_frames, self.embed_dim, 24, 18)

        # reshape model output
        #model_output = model_output.reshape(batch_size*num_frames, -1, self.embed_dim)

        # concatenate intermediate outputs and model output
        intermediate_outputs['model_output'] = model_output
        intermediate_outputs['model_output'] = intermediate_outputs['model_output'].view(batch_size, num_frames, self.embed_dim, 24, 18)
                

        # Feature Fusion
        x = self.feature_fusion(intermediate_outputs)

        # Reshape to separate frames
        x = x.view(batch_size, num_frames, self.embed_dim, 24, 18) # [batch_size, num_frames, 384, 24, 18]  ß

        # Adaptive Frame Weighting
        x, frame_weights = self.adaptive_weighting(x)

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

        if return_weights:
            return x, frame_weights

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
