import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.build import LOSSES


@LOSSES.register_module()
class PointNetLoss(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        # print(cfgs)
        self.mat_diff_loss_scale = cfgs.get('mat_diff_loss_scale', 0.001)

    def feature_transform_reguliarzer(self, trans):
        I = torch.eye(trans.size(1), device=trans.device)[None, :, :]
        loss = torch.mean(torch.norm(trans @ trans.transpose(2, 1) - I, dim=(1, 2)))
        return loss

    def forward(self, pred, target, trans_feat, weight=None):
        loss = F.nll_loss(pred, target) if weight is None else F.nll_loss(pred, target)
        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)
        loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss
