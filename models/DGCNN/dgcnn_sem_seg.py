'''
https://arxiv.org/abs/1801.07829
Dynamic Graph CNN for Learning on Point Clouds
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.DGCNN.dgcnn_utils import *


class DGCNNSemSeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_seg = config.dataset.n_seg
        self.n_block = 3
        self.k_point = config.model.k_point
        self.d_embed = config.model.d_embed
        self.dropout = config.model.dropout
        self.dgcnn_blocks = nn.ModuleList([
            DGCNNBlock([9*2, 64, 64], self.k_point, dim9=True),
            DGCNNBlock([64*2, 64, 64], self.k_point),
            DGCNNBlock([64*2, 64], self.k_point)
        ])
        self.mlp = nn.Sequential(
            PointMLPNd(192, self.d_embed, bias=False, dim=1),
            PointMaxPool(dim=-1, keepdim=True)
        )
        self.head = nn.Sequential(
            PointMLPNd(self.d_embed+64*3, 512, bias=False, dim=1),
            PointMLPNd(512, 256, bias=False, dim=1),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(256, self.n_seg, kernel_size=1, bias=False)
        )

    def forward(self, x):
        _, _, n_point = x.size()                # (batch_size, 9, n_point)
        xs = []
        for i in range(self.n_block):
            x = self.dgcnn_blocks[i](x)         # (batch_size, d_in, n_point) -> (batch_size, d_out, n_point)
            xs.append(x)
        x = torch.cat(xs, dim=1)                # (batch_size, 64*3, n_point)
        x = self.mlp(x).repeat(1, 1, n_point)   # (batch_size, 64*3, n_point) -> (batch_size, d_embed, 1) -> (batch_size, d_embed, n_point)
        x = torch.cat((x, *xs), dim=1)          # (batch_size, d_embed+64*3, n_point)
        x = self.head(x)                        # (batch_size, d_embed+64*3, n_point) -> (batch_size, 13, n_point)
        return x
