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
from torch.autograd import Variable
from models.model_utils import *


class NormAct(nn.Module):
    def __init__(self, d_in, dim, use_act=True):
        super().__init__()
        assert dim in [0, 1, 2]
        self.norm = nn.BatchNorm2d(d_in) if dim == 2 else nn.BatchNorm1d(d_in)
        self.act = nn.LeakyReLU(negative_slope=0.2) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(x))


class PointMLPNd(nn.Module):
    def __init__(self, d_in, d_out, bias=True, dim=1, use_act=True):
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

        self.norm_act = NormAct(d_out, dim=dim, use_act=use_act)

    def forward(self, x):
        return self.norm_act(self.mlp(x))


class PointMaxPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim, keepdim=False)[0]


class PointAvgPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class TransformNet(nn.Module):
    def __init__(self, d_in, d_out_sqrt):
        super().__init__()
        self.d_out_sqrt = d_out_sqrt
        self.mlp = nn.Sequential(
            PointMLPNd(d_in, 64, dim=1),
            PointMLPNd(64, 128, dim=1),
            PointMLPNd(128, 1024, dim=1),
            PointMaxPool(dim=-1),
            PointMLPNd(1024, 512, dim=0),
            PointMLPNd(512, 256, dim=0),
            nn.Linear(256, d_out_sqrt * d_out_sqrt)
        )

    def forward(self, x):
        b_s, _, _ = x.size()
        x = self.mlp(x).view(b_s, self.d_out_sqrt, self.d_out_sqrt)
        iden = Variable(torch.eye(self.d_out_sqrt, device=x.device)).unsqueeze(0).expand(b_s, -1, -1)
        x = x + iden
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super().__init__()
        self.cstn = TransformNet(channel, 3)
        self.mlp1 = PointMLPNd(channel, 64, dim=1)
        self.mlp2 = nn.Sequential(
            PointMLPNd(64, 128, dim=1),
            PointMLPNd(128, 1024, dim=1),
            PointMaxPool(dim=-1)
        )
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.fstn = TransformNet(64, 64) if self.feature_transform else None
   

    def forward(self, x):
        B, D, N = x.size()
        trans = self.cstn(x)

        if D > 3:
            x, feature = x[:, :3, :], x[:, 3:, :]
        x = trans.transpose(2, 1) @ x
        if D > 3:
            x = torch.cat([x, feature], dim=1)

        x = self.mlp1(x)
        trans_feat = self.fstn(x).transpose(2, 1) @ x if self.feature_transform else None
        pointfeat = x
        x = self.mlp2(x)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
