import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2.pointnet2_utils import *


class PointNet2ClsMsg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.use_normals = config.model.use_normals
        d_in = 3 if self.use_normals else 0
        self.sa1 = PointNetSetAbstractionMsg(n_group=512, k_group_list=[16, 32, 128], radius_list=[0.1, 0.2, 0.4], d_features_list=[[d_in+3, 32, 32, 64], [d_in+3, 64, 64, 128], [d_in+3, 64, 96, 128]])
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

    def forward(self, coord):
        feature = coord[:, 3:, :] if self.use_normals else None
        coord = coord[:, :3, :] if self.use_normals else coord

        feature, coord = self.sa1(feature, coord)
        feature, coord = self.sa2(feature, coord)
        feature, coord = self.sa3(feature, coord)
        feature = feature.view(-1, 1024)
        feature = self.head(feature)

        return feature, coord

