import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.abspath('/home/pace/Poseidon/'))

import torch
import torch.nn as nn
import numpy as np
from datasets.zoo.posetrack.PoseTrack import PoseTrack
from posetimation import get_cfg, update_config 
from engine.defaults import default_parse_args
import torchvision.models as models
from torch.utils.data import DataLoader
from mmpose.evaluation.functional import keypoint_pck_accuracy
from easydict import EasyDict


class Poseidon(nn.Module):
    def __init__(self, cfg, device='cpu', phase=TRAIN_PHASE, num_heads=5, embed_dim_for_joint=30):
        super(Poseidon, self).__init__()
        self.device = device
        self.backbone_model = Backbones(cfg, self.device)
        self.backbone = self.backbone_model.backbone.to(self.device)

        # Get heatmap size
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE  # (96, 72)

        # Get model info
        num_params, output_size = self.backbone_model.get_model_info()
        print(f"Backbone parameters: {round(num_params/1e6,1)} M, Output size: {output_size}")

        self.num_heads = num_heads
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.embed_dim_for_joint = embed_dim_for_joint
        self.embed_dim = self.num_joints * self.embed_dim_for_joint

        # Ensure embed_dim is divisible by num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # Fully connected layer for processing backbone output
        self.fc = nn.Linear(output_size, self.embed_dim)

        # Self-attention layer for joints
        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim_for_joint, num_heads=self.num_heads, batch_first=True)

        # Cross-attention layer for frames
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

        # Deconv layers for heatmap generation
        self.deconv_layers = self._make_deconv_layers()

        # Final predictor layer
        self.final_layer = nn.Conv2d(in_channels=self.num_joints, out_channels=self.num_joints, kernel_size=1, stride=1, padding=0)

        self.is_train = True if phase == TRAIN_PHASE else False

        # Print number of parameters
        print(f"Poseidon parameters: {round(self.number_of_parameters(), 1)} M\n\n")

    def _make_deconv_layers(self):
        deconv_layers = []
        input_channels = self.embed_dim
        # Define the configurations for each deconv layer to achieve the desired output size
        # start from [1 ,1] 
        layer_configs = [
            (256, 4, 2, 1),  # [1, 1] -> [2, 2]
            (128, 4, 2, 1),  # [2, 2] -> [4, 4]
            (64, 4, 2, 1),   # [4, 4] -> [8, 8]
            (32, 4, 2, 1),   # [8, 8] -> [16, 16]
            (self.num_joints, (4, 3), (4, 3), (0, 0))  # [16, 16] -> [64, 48]
        ]

        for config in layer_configs:
            if len(config) == 4:
                out_channels, kernel_size, stride, padding = config
                output_padding = 0
            else:
                out_channels, kernel_size, stride, padding, output_padding = config

            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            deconv_layers.append(nn.BatchNorm2d(out_channels))
            deconv_layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels

        return nn.Sequential(*deconv_layers)

    def forward(self, x, meta=None):
        batch_size, num_frames, C, H, W = x.shape
        # print(f"Input shape: {x.shape}")

        x = x.view(-1, C, H, W)
        processed_video = self.backbone(x)
        # print(f"Processed video shape: {processed_video.shape}")

        x = self.fc(processed_video)
        x = x.view(batch_size, num_frames, self.num_joints, self.embed_dim_for_joint)
        # print(f"Shape after FC: {x.shape}")

        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, self.num_joints, self.embed_dim_for_joint)
        x, _ = self.self_attention(x, x, x)
        x = x.view(batch_size, num_frames, self.num_joints, self.embed_dim_for_joint)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, num_frames, self.embed_dim)
        # print(f"Shape after self-attention: {x.shape}")

        central_frame = x[:, num_frames // 2, :].unsqueeze(1)
        context_frames = x
        x, _ = self.cross_attention(central_frame, context_frames, context_frames)
        x = x.squeeze(1)
        x = x.view(batch_size, self.embed_dim, 1, 1)
        # print(f"Before deconv layers: {x.shape}")

        x = self.deconv_layers(x)
        # print(f"After deconv layers: {x.shape}")

        heatmaps = self.final_layer(x)
        # print(f"Final heatmaps shape: {heatmaps.shape}")

        return heatmaps

        print("Accuracy: ", acc)

        return output

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



def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)

    return cfg

def test_model():
    
    args = default_parse_args()
    cfg = setup(args)

    #config_path = '/home/pace/Poseidon/configs/configDCPose.yaml'
    #cfg = load_config(config_path)
    phase = 'train'
    
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    # Initialize the PoseTrack dataset.
    pose_track_dataset = PoseTrack(cfg, phase)

    # Load the backbone model
    backbone = models.resnet50(weights="ResNet50_Weights.DEFAULT").to(device)

    # Create the model
    model = Poseidon(cfg=cfg, device=device, backbone=backbone, phase="train", num_heads=5).to(device)

    # Create the dataloader
    dataloader = DataLoader(pose_track_dataset, batch_size=2, shuffle=True)

    # Get the first batch [input_prev, input_x, input_next, meta]
    input_prev, input_x, input_next, meta = next(iter(dataloader))

    # Send the input to the device
    input_prev = input_prev.to(device)
    input_x = input_x.to(device)
    input_next = input_next.to(device)

    # Concatenate the input with batch dimension
    batch = torch.stack([input_prev, input_x, input_next], dim=1)

    output = model(batch, meta)

    # get the image tensor
    # print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_model()