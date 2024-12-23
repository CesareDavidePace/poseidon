import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, effective_num_joints: int = None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def get_loss_function(cfg, device):
    if cfg.LOSS.NAME == 'JointsMSELoss':
        loss_fn = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    else:
        raise ValueError(f"Unknown loss function: {cfg.LOSS.NAME}")
    
    return loss_fn.to(device)
