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
from models.PointNet.pointnet_utils import *
from models.build import MODELS


@MODELS.register_module()
class PointNetPartSeg(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.n_seg = cfgs.n_seg
        self.stn = TransformNet(3, 3)
        self.mlp1 = PointMLPNd(3, 64, dim=1)
        self.mlp2 = PointMLPNd(64, 128, dim=1)
        self.mlp3 = PointMLPNd(128, 128, dim=1)
        self.mlp4 = PointMLPNd(128, 512, dim=1)
        self.mlp5 = PointMLPNd(512, 2048, dim=1, use_act=False)
        self.fstn = TransformNet(128, 128)
        self.head = nn.Sequential(
            PointMLPNd(4944, 256, dim=1),
            PointMLPNd(256, 256, dim=1),
            PointMLPNd(256, 128, dim=1),
            nn.Conv1d(128, self.n_seg, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, l):
        B, D, N = x.size()
        trans = self.stn(x)

        if D > 3:
            x, f = x[:, :3, :], x[:, 3:, :]
        x = trans.transpose(2, 1) @ x
        if D > 3:
            x = torch.cat([x, f], dim=1)

        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)
        trans_feat = self.fstn(x3)
        x = self.fstn(x3).transpose(2, 1) @ x3
        x4 = self.mlp4(x)
        x5 = self.mlp5(x4)
        x6 = x5.max(dim=-1, keepdim=False)[0]
        x6 = torch.cat([x6, l], dim=1)
        x6 = x6.view(-1, 2048+16, 1).repeat(1, 1, N)
        x = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        x = self.head(x)
        return x, {'trans_feat': trans_feat}
