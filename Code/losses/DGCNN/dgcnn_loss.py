import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.build import LOSSES


@LOSSES.register_module()
class DGCNNLoss(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.smoothing = cfgs.smoothing

    def forward(self, pred, target):
        target = target.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')

        return loss