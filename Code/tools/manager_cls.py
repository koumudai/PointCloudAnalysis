import os
import torch
import time
import tqdm
import torch.nn as nn

from tools import builder
from utils.logger import *

# from torch.optim.lr_scheduler import StepLR

# import utils.data_loaders
# import utils.helpers
# from utils.average_meter import AverageMeter
# from utils.metrics import Metrics
# from utils.schedular import GradualWarmupScheduler
# from utils.loss_utils import get_loss
# from utils.ply import read_ply, write_ply
# import pointnet_utils.pc_util as pc_util
# from PIL import Image


class ManagerCls:
    def __init__(self, args, cfgs):
        print('__init__')
        self.args = args
        self.cfgs = cfgs
        self.logger = get_logger(self.args.log_name)
        # build metric
        self.build_param()
        # build dataset
        # self.build_dataset()
        # build model
        self.build_model()
        # build loss
        self.build_loss()
        # build optimizer
        self.build_optimizer()
        # build scheduler
        self.build_scheduler()
        # build record file
        self.build_record()
        # bulid transform
        self.build_transform()
        # resume model
        self.resume_model()
        # resume optimizer
        self.resume_optimizer()
        # resume scheduler()
        self.resmue_scheduler()
        pass

    def build_param(self):
        self.start_epoch = 0
        # self.best_metrics = Acc_Metric(0.)
        # self.metrics = Acc_Metric(0.)

    def build_dataset(self):
        if self.args.test:
            _, self.test_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.test)
        else:
            self.train_sampler, self.train_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.train)
            _, self.valid_dataloader = builder.build_dataset(self.args, self.cfgs.dataset.valid)

    def build_model(self):
        # build model
        self.model = builder.build_model(self.cfgs.model)

        # load model parameters
        if self.args.test: # test
            builder.load_model(self.model, self.args.ckpts, logger=self.logger)
        else: # train
            # resume ckpts
            if self.args.resume:
                self.start_epoch, self.best_metric = builder.resume_model(self.model, self.args, logger=self.logger)
                self.best_metrics = Acc_Metric(self.best_metrics)
            # TODO: pretrain & finetune is difference
            else:
                if self.args.ckpts is not None:
                    self.model.load_model_from_ckpt(self.args.ckpts)
                else:
                    print_log('Training from scratch', logger=self.logger)

        # to gpu
        if self.args.use_gpu:
            print_log('Using Data parallel ...' , logger=self.logger)
            self.model = nn.DataParallel(self.model).cuda()

    def build_loss(self):
        self.loss = builder.build_loss(self.cfgs.loss)

    def build_optimizer(self):
        self.optimizer = builder.build_optimizer(self.cfgs.optimizer, self.model)

    def build_scheduler(self):
        self.scheduler = builder.build_scheduler(self.cfgs.optimizer, self.model, self.optimizer)

    def build_record(self):
        raise NotImplementedError()

    def build_transform(self):
        raise NotImplementedError()

    def resume_model(self):
        raise NotImplementedError()

    def resume_optimizer(self):
        raise NotImplementedError()

    def resume_scheduler(self):
        raise NotImplementedError()

    def train_net(self):

        start_epoch, end_epoch = 0, self.cfgs.max_epoch

        self.model.zero_grad()

        for epoch_idx in range(start_epoch, end_epoch + 1):
            # train one epoch
            self.train_one_epoch(epoch_idx)
            # valid one epoch
            self.valid_one_epoch(epoch_idx)
            # save checkpoints
            self.save_ckpt()
            pass

    def test_net(self):
        raise NotImplementedError()

    def train_one_epoch(self, epoch_idx):
        epoch_start_time = time.time()
        batch_start_time = time.time()

        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses = AverageMeter(['Loss'])

        # Update learning rate
        self.scheduler.step()

        self.model.train()  # set model to train mode

        for batch_idx, (points, target) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

            assert points.size(1) == self.cfgs.n_point
            points = self.train_transforms(points)

            if self.args.use_gpu:
                points, target = points.cuda(), target.cuda()

            pred = self.model(points)

            loss = self.loss(pred, target.long())

            # TODO: cal metrics

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        epoch_end_time = time.time()

    def valid_one_epoch(self, epoch_idx):

        self.model.eval()  # set model to eval mode

        # TODO: init metrics

        with torch.no_grad():
            for batch_idx, (points, target) in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader)):

                if self.args.use_gpu:
                    points, target = points.cuda(), target.cuda()

                pred = self.model(points)

                # TODO: cal metrics

        # TODO: return metrics
        pass

    def test_one_epoch(self, epoch_idx):
        print('_test_each_epoch')
        pass

    def save_ckpt(self):
        pass


if __name__ == '__main__':
    x = ManagerCls()