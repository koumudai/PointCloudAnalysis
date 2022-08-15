'''
https://arxiv.org/abs/2012.09164
Point Transformer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointcloud_utils import *


class SwapAxes(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(self.dims).contiguous()


class PointConvBNxd(nn.Module):
    def __init__(self, d_in, d_out, dim=2):
        super().__init__()
        if dim == 1:
            self.mlp = nn.Sequential(
                nn.Conv1d(d_in, d_out, kernel_size=1),
                nn.BatchNorm1d(d_out),
                nn.ReLU(),
            )
        elif dim == 2:
            self.mlp = nn.Sequential(
                nn.Conv2d(d_in, d_out, kernel_size=1),
                nn.BatchNorm2d(d_out),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError()
            

    def forward(self, feature):
        return self.mlp(feature)


class PointMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, feature):
        return self.mlp(feature)


class TransitionDown(nn.Module):
    def __init__(self, d_features, n_group, k_group):
        super().__init__()
        self.n_group = n_group
        self.k_group = k_group
        self.mlp = nn.Sequential(
            SwapAxes((0, 3, 1, 2)),
            *[PointConvBNxd(d_in, d_out, dim=2) for d_in, d_out in zip(d_features[:-1], d_features[1:])],
            SwapAxes((0, 2, 3, 1))
        )

    def group(self, feature, coord):
        '''
        Input:
            feature         : batch_size, n_point, d_feature
            coord           : batch_size, n_point, d_coord
        Hidden:
            center_idx      : batch_size, n_group
            center_coord    : batch_size, n_group, d_coord
            neigh_idx       : batch_size, n_group, k_group
            neigh_coord     : batch_size, n_group, k_group, d_coord
            neigh_feature   : batch_size, n_group, k_group, d_coord+d_feature
        Output:
            neigh_feature   : batch_size, n_group, k_group, d_coord+d_feature
            center_coord    : batch_size, n_group, d_coord
        '''
        center_idx = farthest_point_sample(coord, self.n_group)
        center_coord = index_points(coord, center_idx)
        neigh_idx = knn_points(coord, center_coord, self.k_group)
        neigh_coord = index_points(coord, neigh_idx) - center_coord[:, :, None]
        neigh_feature = torch.cat([neigh_coord, index_points(feature, neigh_idx)], dim=-1)
        return neigh_feature, center_coord

    def forward(self, feature, coord):
        '''
        Input:
            feature         : batch_size, n_point, d_feature
            coord           : batch_size, n_point, d_coord
        Hidden:
            neigh_feature   : batch_size, n_group, k_group, d_feature+d_coord
        Output:
            center_feature  : batch_size, n_group, d_features[-1]
            center_coord    : batch_size, n_group, d_coord
        '''
        neigh_feature, center_coord = self.group(feature, coord)
        center_feature = self.mlp(neigh_feature)
        center_feature = center_feature.max(dim=2, keepdim=False)[0]
        return center_feature, center_coord


class TransitionUp(nn.Module):
    def __init__(self, d_in_low, d_in_high, d_out):
        super().__init__()
        self.mlp_low = nn.Sequential(
            SwapAxes((0, 2, 1)),
            PointConvBNxd(d_in_low, d_out, dim=1),
            SwapAxes((0, 2, 1))
        )
        self.mlp_high = nn.Sequential(
            SwapAxes((0, 2, 1)),
            PointConvBNxd(d_in_high, d_out, dim=1),
            SwapAxes((0, 2, 1))
        )

    def interpolate(self, feature_low, coord_low, coord_high):
        '''
        Input:
            feature_low         : batch_size, n_low, d_out
            coord_low           : batch_size, n_low, d_coord
            coord_high          : batch_size, n_high, d_coord
        Output:
            feature_interpolate : batch_size, n_high, d_out
        '''
        dists = square_distance(coord_high, coord_low)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
        feature_interpolate = torch.sum(index_points(feature_low, idx) * weight.unsqueeze(3), dim=2)
        return feature_interpolate

    def forward(self, feature_low, coord_low, feature_high, coord_high):
        '''
        Input:
            feature_low         : batch_size, n_low, d_in_low
            coord_low           : batch_size, n_low, d_coord
            feature_high        : batch_size, n_high, d_in_high
            coord_high          : batch_size, n_high, d_coord
        Output:
            feature_high        : batch_size, n_high, d_out
            coord_high          : batch_size, n_high, d_coord
        '''
        feature_low, feature_high = self.mlp_low(feature_low), self.mlp_high(feature_high)
        feature_high = self.interpolate(feature_low, coord_low, coord_high) + feature_high
        return feature_high, coord_high


class PointTransformerBlock(nn.Module):
    def __init__(self, d_feature, d_coord, d_model, k_point):
        super().__init__()
        self.mlp_1 = nn.Linear(d_feature, d_model)
        self.mlp_2 = nn.Linear(d_model, d_feature)
        self.mlp_gamma = PointMLP(d_model, d_model)
        self.mlp_theta = PointMLP(d_coord, d_model)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.k_point = k_point
        self.scale = d_model ** -0.5

    def forward(self, feature, coord):
        '''
        Input:
            feature         : batch_size, n_point, d_feature
            coord           : batch_size, n_point, d_coord
        Output:
            feature         : batch_size, n_point, d_feature
            coord           : batch_size, n_point, d_coord
        '''
        neigh_idx = knn_points(coord, coord, self.k_point)
        neigh_coord = index_points(coord, neigh_idx)

        pre = feature
        x = self.mlp_1(feature)
        
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), neigh_idx), index_points(self.w_vs(x), neigh_idx)
        pos_enc = self.mlp_theta(coord[:, :, None] - neigh_coord)
        attn = self.mlp_gamma(q[:, :, None] - k + pos_enc) # b x n x k x f
        attn = self.softmax(attn * self.scale)
        feature = torch.einsum('bnkf,bnkf->bnf', attn, v + pos_enc)

        feature = self.mlp_2(feature) + pre
        
        return feature, coord


class PointTransformer(nn.Module):
    def __init__(self, n_point, k_point, d_model, n_block):
        super().__init__()
        self.d_coord = 3
        self.n_block = n_block
        self.n_point = n_point
        self.k_point = k_point
        self.d_model = d_model

        self.mlp_1 = PointMLP(d_in=self.d_coord, d_out=32)
        self.transformer_1 = PointTransformerBlock(32, self.d_coord, self.n_point, self.k_point)

        self.transition_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.n_block):
            d_feature = 32 * 2 ** (i + 1)
            self.transition_blocks.append(TransitionDown([d_feature // 2 + self.d_coord, d_feature], self.n_point // (4 ** (i + 1)), self.k_point))
            self.transformer_blocks.append(PointTransformerBlock(d_feature, self.d_coord, self.d_model, self.k_point))

    def forward(self, x):
        coord = x[..., :self.d_coord] 
        feature, coord = self.transformer_1(self.mlp_1(x), coord)

        feature_and_coord = [(feature, coord)]
        for i in range(self.n_block):
            feature, coord = self.transition_blocks[i](feature, coord)
            feature, coord = self.transformer_blocks[i](feature, coord)
            feature_and_coord.append((feature, coord))
        
        return feature, feature_and_coord
