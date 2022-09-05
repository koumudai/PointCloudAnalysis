'''
Paper: https://arxiv.org/abs/1406.5670
'''
import os
import h5py
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets.build import DATASETS
from utils.logger import *
from datasets.dataset_utils import *


class _ModelNet_XX(Dataset):
    def __init__(self, cfgs, dataset_name, logger):
        super(_ModelNet_XX, self).__init__()
        assert dataset_name in ['ModelNet10', 'ModelNet40'], 'dataset_name error'
        assert logger in ['ModelNet10', 'ModelNet40'], 'logger error'
        self.dataset_path = cfgs.dataset_path
        self.pc_paths = cfgs.pointcloud_path
        self.n_point_all = cfgs.n_point_all
        self.n_class = cfgs.n_class
        self.subset = cfgs.subset
        self.use_normals = cfgs.use_normals
        self.uniform = True
        self.process_data = True
        assert self.subset in ['train', 'test']
        self.class_file = f'{self.dataset_path}/{dataset_name}_shape_names.txt'
        self.class_names = [line.rstrip() for line in open(self.class_file)]
        self.class_dict = dict(zip(self.class_names, range(len(self.class_names))))

        pc_ids = {}
        pc_ids['train'] = [line.rstrip() for line in open(f'{self.dataset_path}/{dataset_name}_train.txt')]
        pc_ids['test'] = [line.rstrip() for line in open(f'{self.dataset_path}/{dataset_name}_test.txt')]

        pc_names = ['_'.join(x.split('_')[0:-1]) for x in pc_ids[self.subset]]
        self.pc_paths = [(pc_name, f'{self.pc_paths}/{pc_name}/{pc_id}.txt') for pc_name, pc_id in zip(pc_names, pc_ids[self.subset])]
        print_log(f'[DATASET] The size of {self.subset} data is {len(self.pc_paths)}', logger=logger)

        if self.uniform:
            self.save_path = f'{self.dataset_path}/{dataset_name}_{self.subset}_{self.n_point_all}pts_fps.dat'
        else:
            self.save_path = f'{self.dataset_path}/{dataset_name}_{self.subset}_{self.n_point_all}pts.dat'

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log(f'[DATASET] Processing data {self.save_path} (only running in the first time)...', logger=logger)
                self.list_of_points = [None] * len(self.pc_paths)
                self.list_of_labels = [None] * len(self.pc_paths)

                for idx in tqdm(range(len(self.pc_paths)), total=len(self.pc_paths)):
                    fn = self.pc_paths[idx]
                    label = np.array([self.class_dict[fn[0]]]).astype(np.int32)
                    points = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        points = farthest_point_sample(points, self.n_point_all)
                    else:
                        points = points[0:self.n_point_all, :]

                    self.list_of_points[idx] = points
                    self.list_of_labels[idx] = label

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log(f'[DATASET] Load processed data from {self.save_path}...', logger=logger)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        if self.process_data:
            points, label = self.list_of_points[idx], self.list_of_labels[idx]
        else:
            fn = self.pc_paths[idx]
            label = np.array([self.class_dict[fn[0]]]).astype(np.int32)
            points = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                points = farthest_point_sample(points, self.n_point_all)
            else:
                points = points[0:self.n_point_all, :]
                
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # n_point_all
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        points = torch.from_numpy(points[pt_idxs].copy()).float()

        feature, coord, label = points, points[:, :3], label[0]

        return feature, coord, label


@DATASETS.register_module()
class ModelNet10(_ModelNet_XX):
    def __init__(self, config):
        super(ModelNet10, self).__init__(config, 'ModelNet10', 'ModelNet10')


@DATASETS.register_module()
class ModelNet40(_ModelNet_XX):
    def __init__(self, config):
        super(ModelNet40, self).__init__(config, 'ModelNet40', 'ModelNet40')
