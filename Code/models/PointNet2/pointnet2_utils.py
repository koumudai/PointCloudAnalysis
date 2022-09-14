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
from models.model_utils import *


class NormAct(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        assert dim in [0, 1, 2]
        self.norm = nn.BatchNorm2d(d_in) if dim == 2 else nn.BatchNorm1d(d_in)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(x))


class PointMLPNd(nn.Module):
    def __init__(self, d_in, d_out, bias=True, dim=1):
        super().__init__()
        assert dim in [0, 1, 2]
        if dim == 0:
            self.mlp = nn.Linear(d_in, d_out, bias=bias)
        elif dim == 1:
            self.mlp = nn.Conv1d(d_in, d_out, kernel_size=1, bias=bias)
        elif dim == 2:
            self.mlp = nn.Conv2d(d_in, d_out, kernel_size=1, bias=bias)
        else:
            raise NotImplementedError()

        self.norm_act = NormAct(d_out, dim=dim)

    def forward(self, x):
        return self.norm_act(self.mlp(x))


class PointMaxPool(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]


class PointAvgPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class PointNetSetAbstractionSsg(nn.Module):
    def __init__(self, n_group, k_group, radius, d_features, group_all=False):
        super().__init__()
        self.n_group = n_group
        self.k_group = k_group
        self.radius = radius
        self.group_all = group_all

        block = []
        for d_in, d_out in zip(d_features[:-1], d_features[1:]):
            block.append(PointMLPNd(d_in, d_out, dim=2))
        block.append(PointMaxPool(dim=-1))
        self.mlp = nn.Sequential(*block)

    def forward(self, x, z):
        """
        Input:
            feature     : point cloud feature data,             [batch_size, d_in, n_point]
            coord       : point cloud coordinate data,          [batch_size, d_coord, n_point]
        Output:
            feature     : sampled point cloud feature data,     [batch_size, d_out, n_group]
            coord       : sampled point cloud coordinate data,  [batch_size, d_coord, n_group]
        """
        x = None if x is None else x.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        x, z = pointnet_sampling_all(x, z) if self.group_all else pointnet_sampling_ball(x, z, self.n_group, self.k_group, self.radius)
        x = self.mlp(x.permute(0, 3, 1, 2))
        z = z.permute(0, 2, 1)
        return x, z


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, n_group, k_group_list, radius_list, d_features_list):
        super().__init__()
        self.n_group = n_group
        self.k_group_list = k_group_list
        self.radius_list = radius_list
        self.mlp_blocks = nn.ModuleList()
        for d_features in d_features_list:
            block = []
            for d_in, d_out in zip(d_features[:-1], d_features[1:]):
                block.append(PointMLPNd(d_in, d_out, dim=2))
            block.append(PointMaxPool(dim=-1))
            self.mlp_blocks.append(nn.Sequential(*block))

    def forward(self, feature, coord):
        """
        Input:
            feature     : point cloud feature data,             [batch_size, d_in, n_point]
            coord       : point cloud coordinate data,          [batch_size, d_coord, n_point]
        Output:
            feature     : sampled point cloud feature data,     [batch_size, d_out, n_group]
            coord       : sampled point cloud coordinate data,  [batch_size, d_coord, n_group]
        """
        coord = coord.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        n_g = self.n_group
        coord_ce = index_points(coord, farthest_point_sample(coord, n_g))
        feature_list = []
        for r, k_g, mlp in zip(self.radius_list, self.k_group_list, self.mlp_blocks):
            idx_ne = ball_points(coord, coord_ce, k_g, r)
            coord_ne = index_points(coord, idx_ne)
            coord_ne -= coord_ce.unsqueeze(2)
            feature_ne = coord_ne if feature is None else torch.cat([index_points(feature, idx_ne), coord_ne], dim=-1)
            feature_ne = feature_ne.permute(0, 3, 1, 2)
            feature_list.append(mlp(feature_ne))

        feature = torch.cat(feature_list, dim=1)
        coord = coord_ce.permute(0, 2, 1)
        return feature, coord


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, d_features):
        super().__init__()
        block = []
        for d_in, d_out in zip(d_features[:-1], d_features[1:]):
            block.append(PointMLPNd(d_in, d_out, dim=1))
        self.mlp = nn.Sequential(*block)

    def forward(self, feature1, coord1, feature2, coord2):
        """
        Input:
            feature1    : point cloud feature data,             [batch_size, d_in_1, n_point]
            coord1      : point cloud coordinate data,          [batch_size, d_coord, n_point]
            feature2    : sampled point cloud feature data,     [batch_size, d_in_2, n_group]
            coord2      : sampled point cloud coordinate data,  [batch_size, d_coord, n_group]
        Output:
            feature     : upsampled feature data,               [batch_size, d_out, n_point]
        """
        coord1 = coord1.permute(0, 2, 1)
        coord2 = coord2.permute(0, 2, 1)

        feature2 = feature2.permute(0, 2, 1)
        n_p, n_g = coord1.size(1), coord2.size(1)

        if n_g == 1:
            interpolated_feature = feature2.repeat(1, n_p, 1)
        else:
            dists = square_distance(coord1, coord2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feature = torch.sum(index_points(feature2, idx) * weight.unsqueeze(3), dim=2)

        if feature1 is not None:
            feature1 = feature1.permute(0, 2, 1)
            feature = torch.cat([feature1, interpolated_feature], dim=-1)
        else:
            feature = interpolated_feature

        feature = self.mlp(feature.permute(0, 2, 1))
        return feature
