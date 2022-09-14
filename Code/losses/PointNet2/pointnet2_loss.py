import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.build import LOSSES


@LOSSES.register_module()
class PointNet2Loss(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

    def forward(self, pred, target, weight=None):
        loss = F.nll_loss(pred, target, weight=weight)
        return loss
