'''
https://arxiv.org/abs/1612.00593
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet.pointnet_utils import *


class PointNetCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.use_normals = config.model.use_normals
        channel = 6 if self.use_normals else 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.n_class),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.head(x)
        return x, trans_feat

