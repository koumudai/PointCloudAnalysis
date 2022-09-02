'''
https://arxiv.org/abs/1801.07829
Dynamic Graph CNN for Learning on Point Clouds
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.DGCNN.dgcnn_utils import *
from models.build import MODELS


@MODELS.register_module()
class DGCNNPartSeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_seg = config.dataset.n_seg
        self.n_cls = config.dataset.n_cls
        self.n_block = 3
        self.k_point = config.model.k_point
        self.d_embed = config.model.d_embed
        self.dropout = config.model.dropout

        self.transform_net = TransformNet()
        self.dgcnn_blocks = nn.ModuleList([
            DGCNNBlock([3*2, 64, 64], self.k_point),
            DGCNNBlock([64*2, 64, 64], self.k_point),
            DGCNNBlock([64*2, 64], self.k_point)
        ])
        self.mlp1 = nn.Sequential(
            PointMLPNd(192, self.d_embed, bias=False, dim=1),
            PointMaxPool(dim=-1, keepdim=True)
        )
        self.mlp2 = PointMLPNd(self.n_cls, 64, bias=False, dim=1)
        self.head = nn.Sequential(
            PointMLPNd(self.d_embed+64*4, 256, bias=False, dim=1),
            nn.Dropout(p=self.dropout),
            PointMLPNd(256, 256, bias=False, dim=1),
            nn.Dropout(p=self.dropout),
            PointMLPNd(256, 128, bias=False, dim=1),
            nn.Conv1d(128, self.n_seg, kernel_size=1, bias=False),
        )

    def forward(self, x, l):
        _, _, n_point = x.size()                            # (batch_size, 9, n_point)
        x0 = get_graph_feature(x, k_point=self.k_point)           # (batch_size, 3, n_point) -> (batch_size, 3*2, n_point, k_point)
        x = self.transform_net(x0).transpose(2, 1) @ x      # (batch_size, 3, 3).T @ (batch_size, 3, n_point) -> (batch_size, 3, n_point)
        xs = []
        for i in range(self.n_block):
            x = self.dgcnn_blocks[i](x)                     # (batch_size, d_in, n_point) -> (batch_size, d_out, n_point)
            xs.append(x)
        x = torch.cat(xs, dim=1)                            # (batch_size, 64*3, n_point)
        x = self.mlp1(x)                                    # (batch_size, 64*3, n_point) -> (batch_size, d_embed, 1)
        l = self.mlp2(l.unsqueeze(2))                       # (batch_size, n_categoties) -> (batch_size, 64, 1)
        x = torch.cat((x, l), dim=1).repeat(1, 1, n_point)  # (batch_size, d_embed+64, n_point)
        x = torch.cat((x, *xs), dim=1)                      # (batch_size, d_embed+64*4, n_point)
        x = self.head(x)                                    # (batch_size, d_embed+64*4, n_point) -> (batch_size, seg_num_all, n_point)
        return x