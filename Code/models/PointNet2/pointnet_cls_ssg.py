'''
Paper Name                  : PointNet: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
Arxiv                       : https://arxiv.org/abs/1706.02413
Official Implementation     : https://github.com/charlesq34/pointnet2
Third Party Implementation  : https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Third Party Implementation  : https://github.com/koumudai/PointCloudAnalysis/tree/master/Code/models/PointNet2
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2.pointnet2_utils import *
from models.build import MODELS


@MODELS.register_module()
class PointNet2ClsSsg(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.n_class = cfgs.n_class
        self.use_normals = cfgs.use_normals
        self.sa1 = PointNetSetAbstractionSsg(n_group=512, k_group=32, radius=0.2, d_features=[3+3, 64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionSsg(n_group=128, k_group=64, radius=0.4, d_features=[128+3, 128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstractionSsg(n_group=None, k_group=None, radius=None, d_features=[256+3, 256, 512, 1024], group_all=True)
        self.head = nn.Sequential(
            PointMLPNd(1024, 512, dim=0),
            nn.Dropout(p=0.4),
            PointMLPNd(512, 256, dim=0),
            nn.Dropout(p=0.4),
            nn.Linear(256, self.n_class),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        z = x[:, :3, :]
        x, z = self.sa1(x, z)
        x, z = self.sa2(x, z)
        x, z = self.sa3(x, z)
        x = x.view(-1, 1024)
        x = self.head(x)

        return x, {}
