import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(coord, n_group):
    '''
    Input:
        coord       : point cloud coord,            [batch_size, n_point, d_coord]
        n_group     : number of samples
    Output:
        center_idx  : sampled point cloud index,    [batch_size, n_group]
    '''
    device = coord.device
    b_s, n_p, d_c, n_g = *coord.shape, n_group
    center_idx = torch.zeros(b_s, n_g, dtype=torch.long, device=device)
    distance = torch.full((b_s, n_p), 1e10, device=device)
    farthest = torch.randint(0, n_p, (b_s, ), dtype=torch.long, device=device)
    batch_idx = torch.arange(b_s, dtype=torch.long, device=device)
    for i in range(n_g):
        center_idx[:, i] = farthest
        centroid = coord[batch_idx, farthest, :].view(b_s, 1, d_c)
        dist = torch.sum((coord - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return center_idx


def square_distance(coord_1, coord_2):
    '''
    Input:
        coord_1     : point cloud coord,    [batch_size, n_point_1, d_coord]
        coord_2     : point cloud coord,    [batch_size, n_point_2, d_coord]
    Output:
        dists       : distance,             [batch_size, n_point_1, n_point_2]
    '''
    dists = torch.sum((coord_1[:, :, None] - coord_2[:, None]) ** 2, dim=-1)
    return dists


def knn_points(coord_all, coord_group, k_group):
    '''
    Input:
        coord_all   : all point cloud coord,            [batch_size, n_point, d_coord]
        coord_group : grouped point cloud coord,        [batch_size, n_group, d_coord]
        k_group     : max sample number in local region
    Output:
        group_idx   : grouped points index,             [batch_size, n_group, k_group]
    '''
    dists = square_distance(coord_group, coord_all)
    group_idx = dists.argsort()[:, :, :k_group]
    return group_idx


def ball_points(coord_all, coord_group, k_group, radius):
    '''
    Input:
        coord_all   : all point cloud coord,            [batch_size, n_point, d_coord]
        coord_group : grouped point cloud coord,        [batch_size, n_group, d_coord]
        k_group     : max sample number in local region
        radius      : local region radius
    Output:
        group_idx   : grouped points index,             [batch_size, n_group, k_group]
    '''
    b_s, n_p, _ = coord_all.shape
    _, n_g, _ = coord_group.shape
    group_idx = torch.arange(n_p, dtype=torch.long, device=coord_all.device).view(1, 1, n_p).repeat([b_s, n_g, 1])
    sqrdists = square_distance(coord_group, coord_all)
    group_idx[sqrdists > radius ** 2] = n_p
    group_idx = group_idx.sort(dim=-1)[0][:, :, :k_group]
    group_first = group_idx[:, :, 0].view(b_s, n_g, 1).repeat([1, 1, k_group])
    mask = group_idx == n_p
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(feature, idx):
    '''
    Input:
        feature     : input point cloud data,   [batch_size, batch_size, d_feature]
        idx         : grouped points index,     [batch_size, n_group, [k_query]]
    Output:
        feature     : indexed point cloud data, [batch_size, n_group, [k_query], d_feature]
    '''
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(feature, 1, idx[..., None].expand(-1, -1, feature.size(-1)))
    return res.reshape(*raw_size, -1)


def pointnet_sampling_knn(feature, coord, n_group, k_group):
    '''
    Input:
        feature     : point cloud feature data,             [batch_size, n_point, d_feature]
        coord       : point cloud coordinate data,          [batch_size, n_point, d_coord]
        n_group     : number of fps point cloud data
        k_group     : number of knn sampling
    Output:
        feature_ne  : sampled point cloud feature data,     [batch_size, n_group, k_group, d_feature+d_coord]
        coord_ce    : sampled point cloud coordinate data,  [batch_size, n_group, d_coord]
    '''
    idx_ce = farthest_point_sample(coord, n_group)
    coord_ce = index_points(coord, idx_ce)
    idx_ne = knn_points(coord, coord_ce, k_group)
    coord_ne = index_points(coord, idx_ne)
    coord_ne = coord_ne - coord_ce.unsqueeze(2)
    feature_ne = coord_ne if feature is None else torch.cat([index_points(feature, idx_ne), coord_ne], dim=-1)
    return feature_ne, coord_ce


def pointnet_sampling_ball(feature, coord, n_group, k_group, radius):
    '''
    Input:
        feature     : point cloud feature data,             [batch_size, n_point, d_feature]
        coord       : point cloud coordinate data,          [batch_size, n_point, d_coord]
        n_group     : number of fps point cloud data
        k_group     : max sample number in local region
        radius      : local region radius
    Output:
        feature_ne  : sampled point cloud feature data,     [batch_size, n_group, k_group, d_feature+d_coord]
        coord_ce    : sampled point cloud coordinate data,  [batch_size, n_group, d_coord]
    '''
    idx_ce = farthest_point_sample(coord, n_group)
    coord_ce = index_points(coord, idx_ce)
    idx_ne = ball_points(coord, coord_ce, k_group, radius)
    coord_ne = index_points(coord, idx_ne)
    coord_ne -= coord_ce.unsqueeze(2)
    feature_ne = coord_ne if feature is None else torch.cat([index_points(feature, idx_ne), coord_ne], dim=-1)
    return feature_ne, coord_ce


def pointnet_sampling_all(feature, coord):
    '''
    Input:
        feature     : point cloud feature data,             [batch_size, n_point, d_feature]
        coord       : point cloud coordinate data,          [batch_size, n_point, d_coord]
    Output:
        feature_ne  : sampled point cloud feature data,     [batch_size, 1, n_point, d_feature+d_coord]
        coord_ce    : sampled point cloud coordinate data,  [batch_size, 1, d_coord]
    '''
    b_s, _, d_c = coord.shape
    feature_ne = coord.unsqueeze(1) if feature is None else torch.cat([feature.unsqueeze(1), coord.unsqueeze(1)], dim=-1)
    coord_ce = torch.zeros((b_s, 1, d_c), device=coord.device)
    return feature_ne, coord_ce