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
class PointNet2ClsMsg(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.n_class = cfgs.n_class
        self.sa1 = PointNetSetAbstractionMsg(n_group=512, k_group_list=[16, 32, 128], radius_list=[0.1, 0.2, 0.4], d_features_list=[[3+3, 32, 32, 64], [3+3, 64, 64, 128], [3+3, 64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(n_group=128, k_group_list=[32, 64, 128], radius_list=[0.2, 0.4, 0.8], d_features_list=[[320+3, 64, 64, 128], [320+3, 128, 128, 256], [320+3, 128, 128, 256]])
        self.sa3 = PointNetSetAbstractionSsg(n_group=None, k_group=None, radius=None, d_features=[640+3, 256, 512, 1024], group_all=True)
        self.head = nn.Sequential(
            PointMLPNd(1024, 512, dim=0),
            nn.Dropout(p=0.4),
            PointMLPNd(512, 256, dim=0),
            nn.Dropout(p=0.5),
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

