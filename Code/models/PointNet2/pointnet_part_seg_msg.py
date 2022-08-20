import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2.pointnet2_utils import *


class PointNet2PartSegMsg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.use_normals = config.model.use_normals
        d_in = 6 if self.use_normals else 3
        self.sa1 = PointNetSetAbstractionMsg(n_group=512, k_group_list=[32, 64, 128], radius_list=[0.1, 0.2, 0.4], d_features_list=[[d_in+3, 32, 32, 64], [d_in+3, 64, 64, 128], [d_in+3, 64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(n_group=128, k_group_list=[64, 128], radius_list=[0.4, 0.8], d_features_list=[[64+128+128+3, 128, 128, 256], [64+128+128+3, 128, 196, 256]])
        self.sa3 = PointNetSetAbstractionSsg(n_group=None, k_group=None, radius=None, d_features=[512+3, 256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(d_features=[256+256+1024, 256, 256])
        self.fp2 = PointNetFeaturePropagation(d_features=[64+128+128+256, 256, 128])
        self.fp1 = PointNetFeaturePropagation(d_features=[16+d_in+3+128, 128, 128])
        self.head = nn.Sequential(
            PointMLPNd(128, 128, dim=1),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.n_class, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, coord, label):
        n_p = coord.size(2)
        feature_0 = coord
        coord_0 = coord[:, :3, :] if self.use_normals else coord

        # Set Abstraction layers
        feature_1, coord_1 = self.sa1(feature_0, coord_0)
        feature_2, coord_2 = self.sa2(feature_1, coord_1)
        feature_3, coord_3 = self.sa3(feature_2, coord_2)

        # Feature Propagation layers
        feature_2 = self.fp3(feature_2, coord_2, feature_3, coord_3)
        feature_1 = self.fp2(feature_1, coord_1, feature_2, coord_2)
        label = label.unsqueeze(2).repeat(1, 1, n_p)
        feature_0 = self.fp1(torch.cat([label, coord_0, feature_0], 1), coord_0, feature_1, coord_1)
        # FC layers
        feature = self.head(feature_0).permute(0, 2, 1).contiguous()
        return feature, coord_3
