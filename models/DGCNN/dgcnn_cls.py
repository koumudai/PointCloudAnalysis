'''
Paper Name                  : Dynamic Graph CNN for Learning on Point Clouds
Paper Id                    : https://arxiv.org/abs/1801.07829
Official Implementation     : https://github.com/WangYueFt/dgcnn
Unofficial Implementation   : https://github.com/antao97/dgcnn.pytorch
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.DGCNN.dgcnn_utils import *


class DGCNNCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_cls = config.dataset.n_cls
        self.n_point = config.model.n_point
        self.k_point = config.model.k_point
        self.n_block = config.model.n_block
        self.d_embed = config.model.d_embed
        self.dropout = config.model.dropout

        self.dgcnn_blocks = nn.ModuleList([
            DGCNNBlock([3*2, 64], self.k_point),
            DGCNNBlock([64*2, 64], self.k_point),
            DGCNNBlock([64*2, 128], self.k_point),
            DGCNNBlock([128*2, 256], self.k_point)
        ])
        self.mlp = PointMLPNd(sum([64, 64, 128, 256]), self.d_embed, bias=False, dim=1)
        self.maxpool = PointMaxPool(dim=-1)
        self.avgpool = PointAvgPool(dim=-1)
        self.head = nn.Sequential(
            PointMLPNd(self.d_embed*2, 512, bias=False, dim=0),
            nn.Dropout(self.dropout),
            PointMLPNd(512, 256, dim=0),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.n_cls),
        )

    def forward(self, x):
        _, _, _ = x.size()                                          # (batch_size, 3, n_point)
        xs = []
        for i in range(self.n_block):
            x = self.dgcnn_blocks[i](x)                             # (batch_size, d_in, n_point) -> (batch_size, d_out, n_point)
            xs.append(x)
        x = torch.cat(xs, dim=1)                                    # (batch_size, 64+64+128+512, n_point)
        x = self.mlp(x)                                             # (batch_size, d_embed, n_point)
        x = torch.cat((self.maxpool(x), self.avgpool(x)), dim=1)    # (batch_size, d_embed*2)
        x = self.head(x)                                            # (batch_size, n_class)
        return x
