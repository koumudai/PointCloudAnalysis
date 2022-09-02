'''
https://arxiv.org/abs/1612.00593
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.pointcloud_utils import *


class PointMLPNd(nn.Module):
    def __init__(self, d_in, d_out, bias=True, dim=1, use_act=True):
        super().__init__()
        block = []
        if dim == 0:
            block.extend([
                nn.Linear(d_in, d_out, bias=bias),
                nn.BatchNorm1d(d_out)
            ])
        elif dim == 1:
            block.extend([
                nn.Conv1d(d_in, d_out, kernel_size=1, bias=bias),
                nn.BatchNorm1d(d_out)
            ])
        elif dim == 2:
            block.extend([
                nn.Conv2d(d_in, d_out, kernel_size=1, bias=bias),
                nn.BatchNorm2d(d_out)
            ])
        else:
            raise NotImplementedError()
        if use_act:
            block.append(nn.ReLU())
        self.mlp = nn.Sequential(*block)

    def forward(self, x):
        return self.mlp(x)


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
