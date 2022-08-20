import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointTransformer.pointtransformer_utils import *
from models.build import MODELS


@MODELS.register_module()
class PointTransformerCls(nn.Module):
    def __init__(self, n_class=40, n_point=1024, k_point=16, d_model=512, n_block=4):
        super().__init__()
        self.backbone = PointTransformer(n_point, k_point, d_model, n_block)
        self.mlp_2 = nn.Sequential(
            nn.Linear(32 * 2 ** n_block, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        '''
        Input:
            x           : batch_size, d_coord, n_point
        Output:
            feature     : batch_size, n_class
        '''
        x = x.permute(0, 2, 1).contiguous()
        feature, _ = self.backbone(x)
        feature = self.mlp_2(feature.mean(1))
        return feature


