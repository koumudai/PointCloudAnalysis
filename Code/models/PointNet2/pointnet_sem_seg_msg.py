import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2.pointnet2_utils import *


class PointNet2SemSegMsg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.sa1 = PointNetSetAbstractionMsg(n_group=1024, k_group_list=[16, 32], radius_list=[0.05, 0.1], d_features_list=[[9+3, 16, 16, 32], [9+3, 32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(n_group=256, k_group_list=[16, 32], radius_list=[0.1, 0.2],  d_features_list=[[32+64+3, 64, 64, 128], [32+64+3, 64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(n_group=64, k_group_list=[16, 32], radius_list=[0.2, 0.4], d_features_list=[[128+128+3, 128, 196, 256], [128+128+3, 128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(n_group=16, k_group_list=[16, 32], radius_list=[0.4, 0.8], d_features_list=[[256+256+3, 256, 256, 512], [256+256+3, 256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(d_features=[256+256+512+512, 256, 256])
        self.fp3 = PointNetFeaturePropagation(d_features=[128+128+256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(d_features=[32+64+256, 256, 128])
        self.fp1 = PointNetFeaturePropagation(d_features=[9+128, 128, 128, 128])
        self.head = nn.Sequential(
            PointMLPNd(128, 128, dim=1),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.n_class, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, coord):
        feature_0 = coord
        coord_0 = coord[:, :3, :]

        # Set Abstraction layers
        feature_1, coord_1 = self.sa1(feature_0, coord_0)
        feature_2, coord_2 = self.sa2(feature_1, coord_1)
        feature_3, coord_3 = self.sa3(feature_2, coord_2)
        feature_4, coord_4 = self.sa4(feature_3, coord_3)

        # Feature Propagation layers
        feature_3 = self.fp4(feature_3, coord_3, feature_4, coord_4)
        feature_2 = self.fp3(feature_2, coord_2, feature_3, coord_3)
        feature_1 = self.fp2(feature_1, coord_1, feature_2, coord_2)
        feature_0 = self.fp1(feature_0, coord_0, feature_1, coord_1)

        # FC layers
        feature = self.head(feature_0).permute(0, 2, 1).contiguous()
        return feature, coord_4
