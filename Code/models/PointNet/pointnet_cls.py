'''
Paper Name                  : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Arxiv                       : https://arxiv.org/abs/1612.00593
Official Implementation     : https://github.com/charlesq34/pointnet
Third Party Implementation  : https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Third Party Implementation  : https://github.com/koumudai/PointCloudAnalysis/tree/master/Code/models/PointNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet.pointnet_utils import *
from models.build import MODELS


@MODELS.register_module()
class PointNetCls(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.n_class = cfgs.n_class
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
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
        x, trans, trans_feat = self.encoder(x)
        x = self.head(x)
        return x, {'trans_feat': trans_feat}