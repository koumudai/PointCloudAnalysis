import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.build import LOSSES

def feature_transform_reguliarzer(trans):
    I = torch.eye(trans.size(1), device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(trans @ trans.transpose(2, 1) - I, dim=(1, 2)))
    return loss

@LOSSES.register_module()
class PointNetLoss(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.mat_diff_loss_scale = cfgs.get('mat_diff_loss_scale', 0.001)

    def forward(self, pred, target, **rtkwargs):
        trans_feat = rtkwargs['trans_feat']
        weight = rtkwargs.get('weight', None)
        loss = F.nll_loss(pred, target, weight=weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        # print(loss, mat_diff_loss, self.mat_diff_loss_scale)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
