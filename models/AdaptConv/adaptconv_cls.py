'''
PaperName   : Adaptive Graph Convolution for Point Cloud Analysis
PaperId     : https://arxiv.org/abs/2108.08035
Code        : https://github.com/hrzhou2/AdaptConv-master
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.AdaptConv.adaptconv_utils import *


class AdaptConvCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_class = config.dataset.n_class
        self.k_point = config.model.k_point
        self.d_embed = config.model.d_embed
        self.dropout = config.model.dropout

        self.adapt_conv1 = AdaptiveConv(6, 6, 64)
        self.adapt_conv2 = AdaptiveConv(64*2, 6, 64)
        self.conv3 = PointMLPNd(64*2, 128, bias=False, dim=2)
        self.conv4 = PointMLPNd(128*2, 256, bias=False, dim=2)
        self.conv5 = PointMLPNd(512, self.d_embed, bias=False, dim=1)
        self.maxpool = PointMaxPool(dim=-1)
        self.avgpool = PointAvgPool(dim=-1)
        self.head = nn.Sequential(
            PointMLPNd(self.d_embed*2, 512, bias=False, dim=0),
            nn.Dropout(p=self.dropout),
            PointMLPNd(512, 256, dim=0),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, self.n_class)
        )

    def forward(self, x):
        _, _, _ = x.size()                                              # (batch_size, 3, n_point)
        coord = x[:, :3, :]                                             # (batch_size, 3, n_point)

        x, idx = get_graph_feature(x, k=self.k_point)
        p, _ = get_graph_feature(coord, k=self.k_point, idx=idx)
        x1 = self.adapt_conv1(p, x).max(dim=-1, keepdim=False)[0]

        x, idx = get_graph_feature(x1, k=self.k_point)
        p, _ = get_graph_feature(coord, k=self.k_point, idx=idx)
        x2 = self.adapt_conv2(p, x).max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x2, k=self.k_point)
        x3 = self.conv3(x).max(dim=-1, keepdim=False)[0]

        x, _ = get_graph_feature(x3, k=self.k_point)
        x4 = self.conv4(x).max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x = torch.cat((self.maxpool(x), self.avgpool(x)), dim=1)

        x = self.head(x)
        return x