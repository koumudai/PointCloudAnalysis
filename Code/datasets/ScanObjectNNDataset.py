'''
Paper: https://arxiv.org/abs/1908.04616
Code: https://github.com/hkust-vgd/scanobjectnn
'''
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
from datasets.dataset_utils import *


class _ScanObjectNN_XX(Dataset):
    def __init__(self, config, file_name):
        super().__init__()
        self.dataset_path = config.dataset_path
        self.pc_path = config.pointcloud_path
        self.subset = config.subset
        assert self.subset in ['train', 'test']
        h5 = h5py.File(f'{self.pc_path}/{file_name}', 'r')
        self.points = np.array(h5['data']).astype(np.float32)
        self.labels = np.array(h5['label']).astype(int)
        h5.close()

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # n_point
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        points = self.points[idx, pt_idxs].copy()
        points = torch.from_numpy(points).float()
        label = self.labels[idx]
        return points, label

@DATASETS.register_module()
class ScanObjectNN_OBJ_ONLY(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset.h5'
        super(ScanObjectNN_OBJ_ONLY, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_ONLY shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_ONLY_PB_T25(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmented25_norot.h5'
        super(ScanObjectNN_OBJ_ONLY_PB_T25, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_ONLY_PB_T25 shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_ONLY_PB_T25_R(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmented25rot.h5'
        super(ScanObjectNN_OBJ_ONLY_PB_T25_R, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_ONLY_PB_T25_R shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_ONLY_PB_T50_R(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmentedrot.h5'
        super(ScanObjectNN_OBJ_ONLY_PB_T50_R, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_ONLY_PB_T50_R shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_ONLY_PB_T50_RS(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmentedrot_scale75.h5'
        super(ScanObjectNN_OBJ_ONLY_PB_T50_RS, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_ONLY_PB_T50_RS shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_BG(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset.h5'
        super(ScanObjectNN_OBJ_BG, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_BG shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_BG_PB_T25(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmented25_norot.h5'
        super(ScanObjectNN_OBJ_BG_PB_T25, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_BG_PB_T25 shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_BG_PB_T25_R(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmented25rot.h5'
        super(ScanObjectNN_OBJ_BG_PB_T25_R, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_BG_PB_T25_R shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_BG_PB_T50_R(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmentedrot.h5'
        super(ScanObjectNN_OBJ_BG_PB_T50_R, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_BG_PB_T50_R shape of {self.points.shape}')


@DATASETS.register_module()
class ScanObjectNN_OBJ_BG_PB_T50_RS(_ScanObjectNN_XX):
    def __init__(self, config):
        file_name = ('training' if config.subset is 'train' else 'test') + '_objectdataset_augmentedrot_scale75.h5'
        super(ScanObjectNN_OBJ_BG_PB_T50_RS, self).__init__(config, file_name)
        print(f'Successfully load ScanObjectNN_OBJ_BG_PB_T50_RS shape of {self.points.shape}')
