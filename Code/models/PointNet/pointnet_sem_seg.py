'''
https://arxiv.org/abs/1612.00593
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet.pointnet_utils import *
from models.build import MODELS


@MODELS.register_module()
class PointNetSemSeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.head = nn.Sequential(
            PointMLPNd(1088, 512, dim=1),
            PointMLPNd(512, 256, dim=1),
            PointMLPNd(256, 128, dim=1),
            nn.Conv1d(128, self.n_class, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.head(x)
        x = x.transpose(2,1).contiguous()
        return x, trans_feat
