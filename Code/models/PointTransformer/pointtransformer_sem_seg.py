import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointTransformer.pointtransformer_utils import *

class PointTransformerSeg(nn.Module):
    def __init__(self, n_class=40, n_point=1024, k_point=16, d_model=512, n_block=4):
        super().__init__()
        self.d_coord = 3
        self.n_block = n_block

        self.backbone = PointTransformer(n_point, k_point, d_model, n_block)
        self.mlp_2 = nn.Sequential(
            nn.Linear(32 * 2 ** n_block, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** n_block)
        )
        self.transformer_2 = PointTransformerBlock(32 * 2 ** n_block, self.d_coord, d_model, k_point)

        self.transition_ups = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        for i in reversed(range(n_block)):
            d_feature = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(d_feature*2, d_feature, d_feature))
            self.transformer_blocks.append(PointTransformerBlock(d_feature, self.d_coord, d_model, k_point))

        self.mlp_3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )        


    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        _, feature_and_coord = self.backbone(x)
        feature, coord = feature_and_coord[-1]
        feature, coord = self.transformer_2(self.mlp_2(feature), coord)

        for i in range(self.n_block):
            feature, coord = self.transition_ups[i](feature, coord, *feature_and_coord[-i-2])
            feature, coord = self.transformer_blocks[i](feature, coord)
        
        feature = self.mlp_3(feature)
        return feature
