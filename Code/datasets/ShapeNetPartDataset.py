import os
import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
from datasets.build import DATASETS
from utils.logger import *
from datasets.dataset_utils import *


def download_shapenetpart(pc_path, path_name='shapenetpart_2048'):
    if not os.path.exists(pc_path):
        os.mkdir(pc_path)
    if not os.path.exists(f'{pc_path}/{path_name}'):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system(f'wget {www} --no-check-certificate')
        os.system(f'unzip {zipfile}')
        os.system(f'mv hdf5_data {pc_path}/{path_name}')
        # os.system(f'rm {zipfile}')


def load_shapenetpart(pc_path, subset, path_name='shapenetpart_2048'):
    assert subset in ['train&valid', 'train', 'test']
    download_shapenetpart(pc_path)
    if subset == 'train&valid':
        file_paths = glob.glob(f'{pc_path}/{path_name}/*train*.h5') + glob.glob(f'{pc_path}/{path_name}/*val*.h5')
    elif subset in ['train', 'test']:
        file_paths = glob.glob(f'{pc_path}/{path_name}/*{subset}*.h5')
    else:
        raise NotImplementedError()

    all_data, all_label, all_seg = [], [], []
    for path in file_paths:
        f = h5py.File(path, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


class _ShapeNetPart_XX(Dataset):
    def __init__(self, cfgs, dataset_name, logger):
        assert dataset_name in ['ShapeNetPart'], 'dataset_name error'
        assert logger in ['ShapeNetPart'], 'logger error'
        self.pc_path = cfgs.pointcloud_path
        self.subset = cfgs.subset
        assert self.subset in ['train&valid', 'train', 'test']
        self.data, self.label, self.seg = load_shapenetpart(self.pc_path, self.subset)

        self.n_point = cfgs.n_point
        self.class_choice = cfgs.class_choice
        self.cat2id = cfgs.cat2id
        self.seg_num = cfgs.seg_num
        self.index_start = cfgs.index_start

        if self.class_choice != None:
            print_log(f'[DATASET] Select all data which class is {self.class_choice}', logger=logger)
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            print_log(f'[DATASET] Use all data', logger=logger)
            self.seg_num_all = sum(self.seg_num)
            self.seg_start_index = 0

    def __getitem__(self, item):
        points = self.data[item][:self.n_point]
        label = self.label[item]
        seg = self.seg[item][:self.n_point]
        if self.subset != 'test':
            points = translate_pointcloud(points)
            indices = list(range(points.shape[0]))
            np.random.shuffle(indices)
            points = points[indices]
            seg = seg[indices]
        return points, label, seg

    def __len__(self):
        return self.data.shape[0]


@DATASETS.register_module()
class ShapeNetPart(_ShapeNetPart_XX):
    def __init__(self, cfgs):
        super(ShapeNetPart, self).__init__(cfgs, 'ShapeNetPart', 'ShapeNetPart')


# if __name__ == '__main__':
#     class Config:
#         def __init__(self):
#             self.pointcloud_path = 'data/ShapeNetPart'
#             self.subset = 'test'
    
#     x = ShapeNetPart(Config())
