import os
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn as nn
from abc import abstractmethod, ABCMeta
from managers import builder
from models.model_utils import *
from utils.logger import *
from utils.metrics import *
from utils.misc import cal_model_parm_nums

from tensorboardX import SummaryWriter

class ManagerBase(metaclass=ABCMeta):
    def __init__(self, args, cfgs):
        # build config
        self.build_config(args, cfgs)
        # build dataset
        self.build_dataset()
        # build model
        self.build_model()
        # build loss
        self.build_criterion()
        # build optimizer
        self.build_optimizer()
        # build scheduler
        self.build_scheduler()
        # build tensorboard writer
        self.build_writer()
        # bulid transform
        # self.build_transform()
        # build other information
        self.build_other_info()
        # set experiment
        self.set_experiment()

    def build_config(self, args, cfgs):
        self.args = args
        self.cfgs = cfgs
        self.logger = get_logger(self.args.log_name)

    def build_dataset(self):
        if not self.args.test:
            self.train_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.train)
            self.valid_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.valid)
        else:
            self.test_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.test)

    def build_model(self):
        # build model
        self.model = builder.build_model(self.cfgs.model)
        self.logger.info(f'[Model] Model: \n{self.model}')
        # log model
        model_size = cal_model_parm_nums(self.model)
        self.logger.info(f'[Model] Number of params: {(model_size / 1e6):.4f} M')

        # to gpu
        if self.args.use_gpu:
            self.logger.info('[Model] Using Data parallel ...')
            self.model = nn.DataParallel(self.model).cuda()

    def build_criterion(self):
        self.criterion = builder.build_criterion(self.cfgs.loss)

    def build_optimizer(self):
        self.optimizer = builder.build_optimizer(self.cfgs.optimizer, self.model)

    def build_scheduler(self):
        self.scheduler = builder.build_scheduler(self.cfgs.scheduler, self.optimizer)

    def build_writer(self):
        if not self.args.test:
            self.train_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'train'))
            self.valid_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'valid'))
        else:
            self.test_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'test'))

    def build_transform(self):
        raise NotImplementedError()

    @abstractmethod
    def build_other_info(self):
        raise NotImplementedError()

    def set_experiment(self):
        if self.args.test: # test
            self.load_checkpoint(self.args.ckpt_path)
        elif self.args.resume: # train & resume
            self.cfgs.start_epoch = self.resume_checkpoint(self.args.ckpt_path)
        else: # train from scratch
            self.cfgs.start_epoch = 0

    def save_checkpoint(self, prefix, epoch_idx):
        ckpt_path = f'{self.args.exp_path}/{prefix}.pth'
        builder.save_checkpoint(ckpt_path, epoch_idx, self.model, self.optimizer, self.scheduler, self.logger)

    def load_checkpoint(self, ckpt_path):
        ckpt_path = ckpt_path if ckpt_path is not None else f'{self.args.exp_path}/../train/ckpt-best.pth'
        epoch_idx = builder.load_checkpoint(ckpt_path, self.model, self.logger)
        return epoch_idx

    def resume_checkpoint(self, ckpt_path):
        ckpt_path = ckpt_path if ckpt_path is not None else f'{self.args.exp_path}/../train/ckpt-last.pth'
        epoch_idx = builder.resume_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler, self.logger)
        return epoch_idx

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def train_one_epoch(self, epoch_idx, cfgs, dataloader):
        pass
    
    @abstractmethod
    @torch.no_grad()
    def valid_one_epoch(self, cfgs, dataloader):
        pass

    @abstractmethod
    @torch.no_grad()
    def test_one_epoch(self, cfgs, dataloader):
        pass

    @abstractmethod
    def print_results(self, epoch_idx, metrics):
        pass


if __name__ == '__main__':
    m = ManagerBase(1, 2)
