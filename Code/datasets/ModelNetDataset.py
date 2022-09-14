'''
Paper Name                  : 3D ShapeNets: A Deep Representation for Volumetric Shapes
Arxiv                       : Paper: https://arxiv.org/abs/1406.5670
'''
import os
import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
from datasets.build import DATASETS
from utils.logger import *
from datasets.dataset_utils import *


def download_modelnet40(pc_path, path_name='modelnet40_2048'):
    if not os.path.exists(pc_path):
        os.mkdir(pc_path)
    if not os.path.exists(f'{pc_path}/{path_name}'):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system(f'wget {www} --no-check-certificate')
        os.system(f'unzip {zipfile}')
        os.system(f'mv {zipfile[:-4]} {pc_path}/{path_name}')
        # os.system(f'rm {zipfile}')


def load_modelnet(pc_path, subset, path_name='modelnet40_2048'):
    assert subset in ['train', 'test']
    download_modelnet40(pc_path)
    file_paths = glob.glob(f'{pc_path}/{path_name}/*{subset}*.h5')

    all_data, all_label = [], []
    for path in file_paths:
        f = h5py.File(path, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class _ModelNet_XX(Dataset):
    def __init__(self, cfgs, dataset_name, logger):
        super(_ModelNet_XX, self).__init__()
        assert dataset_name in ['ModelNet10', 'ModelNet40'], 'dataset_name error'
        assert logger in ['ModelNet10', 'ModelNet40'], 'logger error'
        if dataset_name != 'ModelNet40':
            raise NotImplementedError()
        self.pc_path = cfgs.pointcloud_path
        self.subset = cfgs.subset
        assert self.subset in ['train', 'test']
        self.n_point = cfgs.n_point
        self.n_class = cfgs.n_class
        self.class_names = cfgs.cat2id.keys()
        self.data, self.label = load_modelnet(self.pc_path, self.subset)

    def __getitem__(self, item):
        points = self.data[item][:self.n_point]
        label = self.label[item]
        if self.subset == 'train':
            points = translate_pointcloud(points)
            indices = list(range(points.shape[0]))
            np.random.shuffle(indices)
            points = points[indices]
        return points, label

    def __len__(self):
        return self.data.shape[0]


@DATASETS.register_module()
class ModelNet10(_ModelNet_XX):
    def __init__(self, cfgs):
        super(ModelNet10, self).__init__(cfgs, 'ModelNet10', 'ModelNet10')


@DATASETS.register_module()
class ModelNet40(_ModelNet_XX):
    def __init__(self, cfgs):
        super(ModelNet40, self).__init__(cfgs, 'ModelNet40', 'ModelNet40')


# if __name__ == '__main__':
#     class Config:
#         def __init__(self):
#             self.pointcloud_path = 'data/ModelNet40'
#             self.subset = 'test'
    
#     x = ModelNet40(Config())
