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


class PointNet2PartSegSsg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.model.n_class
        self.use_normals = config.model.use_normals
        d_in = 6 if self.use_normals else 3
        self.sa1 = PointNetSetAbstractionSsg(n_group=512, k_group=32, radius=0.2, d_features=[d_in+3, 64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstractionSsg(n_group=128, k_group=64, radius=0.4, d_features=[128+3, 128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstractionSsg(n_group=None, k_group=None, radius=None, d_features=[256+3, 256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(d_features=[256+1024, 256, 256])
        self.fp2 = PointNetFeaturePropagation(d_features=[128+256, 256, 128])
        self.fp1 = PointNetFeaturePropagation(d_features=[16+d_in+3+128, 128, 128, 128])
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
