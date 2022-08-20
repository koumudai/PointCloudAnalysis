'''
https://arxiv.org/abs/1801.07829
Dynamic Graph CNN for Learning on Point Clouds
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.pointcloud_utils import *


def get_graph_feature(feature, k_group, idx=None, dim9=False):
    '''
    Input:
        feature     : point cloud feature,              [batch_size, d_feature, n_point]
        k_group     : max sample number in local region
        idx         : grouped points index,             [batch_size, n_point, k_group]
        dim9        : for seg
    Output:
        feature     : point cloud feature,              [batch_size, 2*d_feature, n_point, k_group]
    '''
    feature = feature.permute(0, 2, 1).contiguous()
    if idx is None:
        if dim9:
            idx = knn_points(feature[..., 6:], feature[..., 6:], k_group)
        else:
            idx = knn_points(feature, feature, k_group)
    feature_ne = index_points(feature, idx)
    feature = feature[:, :, None].expand(-1, -1, k_group, -1)
    feature = torch.cat((feature_ne-feature, feature), dim=3)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature


class NormAct(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        assert dim in [0, 1, 2]
        self.norm = nn.BatchNorm2d(d_in) if dim == 2 else nn.BatchNorm1d(d_in)
        self.act = nn.LeakyReLU(negative_slope=0.2)
    
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


class DGCNNBlock(nn.Module):
    def __init__(self, d_features, k_point, dim9=False):
        super().__init__()
        self.k_point = k_point
        self.dim9 = dim9
        block = []
        for d_in, d_out in zip(d_features[:-1], d_features[1:]):
            block.append(PointMLPNd(d_in, d_out, bias=False, dim=2))
        block.append(PointMaxPool(dim=-1))
        self.mlp = nn.Sequential(*block)

    def forward(self, x):
        x = get_graph_feature(x, self.k_point, dim9=self.dim9)  # b_s, d_in, n_p, k_p
        x = self.mlp(x)  # b_s, d_out, n_p
        return x


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            PointMLPNd(6, 64, bias=False, dim=2),
            PointMLPNd(64, 128, bias=False, dim=2),
            PointMaxPool(dim=-1),
            PointMLPNd(128, 1024, bias=False, dim=1),
            PointMaxPool(dim=-1),
            PointMLPNd(1024, 512, bias=False, dim=0),
            PointMLPNd(512, 256, bias=False, dim=0),
        )
        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        x = self.mlp(x)
        x = self.transform(x).view(-1, 3, 3)
        return x

