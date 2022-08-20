import os
import h5py
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    bound = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / bound
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint, ))
    distance = np.full((N, ), 1e10)
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class _ShapeNet_XX(Dataset):
    def __init__(self, config, dataset_name, logger):
        self.dataset_path = config.dataset_path
        self.pc_path = config.pointcloud_path
        self.n_point = config.n_point
        self.subset = config.subset
        assert self.subset in ['train', 'test']
        self.uniform = True
        self.use_random_sample = True if self.subset is 'train' else False

        self.class_file = f'{self.dataset_path}/{dataset_name}_shape_names.txt'
        self.class_names = [line.rstrip() for line in open(self.class_file)]
        self.class_dict = dict(zip(self.class_names, range(len(self.class_names))))
        self.shapenet_dict = json.load(open(f'{self.dataset_path}/{dataset_name}_synset_dict.json', 'r'))

        pc_ids = {}
        pc_ids['train'] = [line.rstrip() for line in open(f'{self.dataset_path}/{dataset_name}_train.txt')]
        pc_ids['test'] = [line.rstrip() for line in open(f'{self.dataset_path}/{dataset_name}_test.txt')]

        pc_names = [self.shapenet_dict[x.split('-')[0]] for x in pc_ids[self.subset]]
        self.pc_paths = [(pc_name, f'{self.pc_path}/{pc_id}.txt') for pc_name, pc_id in zip(pc_names, pc_ids[self.subset])]

        print_log(f'[DATASET] The size of {self.subset} data is {len(self.pc_path)}', logger=logger)
        print_log(f'[DATASET] {len(self.pc_paths)} instances were loaded', logger=logger)

    def __len__(self):
        return len(self.pc_paths)

    def _get_item(self, idx):
        fn = self.pc_paths[idx]
        label = np.array([self.class_dict[fn[0]]]).astype(np.int32)
        point_set = np.load(fn[1]).astype(np.float32)

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.n_point)
        else:
            point_set = point_set[0:self.n_point, :]

        point_set = pc_normalize(point_set)
        pt_idxs = np.arange(0, point_set.shape[0])   # n_point
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = point_set[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label[0]

    def __getitem__(self, idx):
        points, label = self._get_item(idx)
        return points, label

@DATASETS.register_module()
class ShapeNet55(Dataset):
    def __init__(self, config):
        super(ShapeNet55, self).__init__(config, dataset_name='shapenet55', logger='ShapeNet55')
