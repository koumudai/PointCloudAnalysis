import os
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn as nn

from managers import builder
from utils.logger import *
from utils.metrics import *
from models.model_utils import *

from tensorboardX import SummaryWriter
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
        self.args = args
        self.cfgs = cfgs
        self.logger = get_logger(self.args.log_name)
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
        # set experiment
        self.set_experiment()
        pass

    def build_dataset(self):
        if not self.args.test:
            self.train_dataloader, self.class_names = builder.build_dataset(self.args, self.cfgs.dataset.train)
            self.valid_dataloader, _ = builder.build_dataset(self.args, self.cfgs.dataset.valid)
        else:
            self.test_dataloader, self.class_names = builder.build_dataset(self.args, self.cfgs.dataset.test)

    def build_model(self):
        # build model
        self.model = builder.build_model(self.cfgs.model)
        # log model
        # model_size = cal_model_parm_nums(self.model)
        self.logger.info(self.model)
        # self.logger.info('[Model] Number of params: %.4f M' % (model_size / 1e6))

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

    def set_experiment(self):
        if self.args.test: # test    
            self.load_checkpoint('ckpt-best')
        elif self.args.resume: # train & resume
            self.resume_checkpoint('ckpt-best')
            oa, macc, cls_accs, cm = self.valid_one_epoch(self.cfgs, self.valid_dataloader)
            self.print_cls_results(self.cfgs.start_epoch, oa, macc, cls_accs)
        else: # train from scratch
            self.cfgs.start_epoch = 0

    def save_checkpoint(self, prefix, epoch_idx, metrics):
        ckpt_path = f'{self.args.exp_path}/{prefix}.pth'
        builder.save_checkpoint(ckpt_path, epoch_idx, self.model, self.optimizer, self.scheduler, metrics, self.logger)

    def load_checkpoint(self, prefix):
        ckpt_path = f'{self.args.exp_path}/../train/{prefix}.pth'
        epoch_idx = builder.load_checkpoint(ckpt_path, self.model, self.logger)
        return epoch_idx

    def resume_checkpoint(self, prefix):
        ckpt_path = f'{self.args.exp_path}/../train/{prefix}.pth'
        epoch_idx, metrics = builder.resume_checkpoint(ckpt_path, self.cfgs, self.model, self.optimizer, self.scheduler, self.logger)
        return epoch_idx, metrics

    def print_cls_results(self, epoch, oa, macc, class_accs):
        msg = f'Classes Accuracy\n'
        for name, acc in zip(self.class_names, class_accs):
            msg += f'{name:10}: {acc:3.2f}%\n'
        msg += f'Epoch: {epoch:4}\tOverall Accuracy: {oa:3.2f}\tMean Accuracy: {macc:3.2f}\n'
        self.logger.info(msg)

    def train(self):
        valid_oa, valid_macc, valid_accs, valid_best_oa, valid_macc_when_best_oa, best_epoch = 0., 0., [], 0., 0., 0

        start_epoch, end_epoch = self.cfgs.start_epoch, self.cfgs.max_epoch

        self.model.zero_grad()

        for epoch_idx in range(start_epoch + 1, end_epoch + 1):
            is_best = False

            # train one epoch
            train_loss, train_oa, train_macc, _, _ = self.train_one_epoch(epoch_idx, self.cfgs, self.train_dataloader)

            # valid one epoch
            # if epoch_idx % self.cfg.valid_freq == 0:
            if epoch_idx % 1 == 0:
                valid_oa, valid_macc, valid_accs, _ = self.valid_one_epoch(self.cfgs, self.valid_dataloader)
                is_best = valid_oa > valid_best_oa
                if is_best:
                    valid_best_oa = valid_oa
                    valid_macc_when_best_oa = valid_macc
                    best_epoch = epoch_idx
                    self.logger.info(f'Find a better ckpt @Epoch {epoch_idx}')
                    self.print_cls_results(epoch_idx, valid_oa, valid_macc, valid_accs)
                    self.save_checkpoint('ckpt-best', epoch_idx, metrics={'train_oa': train_oa, 'train_macc': train_macc, 'valid_oa': valid_oa, 'valid_macc': valid_macc})
            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Epoch {epoch_idx}, lr {lr:.6f}, train_oa {train_oa:.2f}, valid_oa {valid_oa:.2f}, valid_best_oa {valid_best_oa:.2f}')
            if self.train_writer is not None:
                self.train_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.train_writer.add_scalar('train_loss', train_loss, epoch_idx)
                self.train_writer.add_scalar('train_oa', train_oa, epoch_idx)
                self.train_writer.add_scalar('train_macc', train_macc, epoch_idx)
            if self.valid_writer is not None:
                self.valid_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.valid_writer.add_scalar('valid_oa', valid_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc', valid_macc, epoch_idx)
                self.valid_writer.add_scalar('valid_best_oa', valid_best_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc_when_best_oa', valid_macc_when_best_oa, epoch_idx)
                self.valid_writer.add_scalar('best_epoch', best_epoch, epoch_idx)

        if self.train_writer is not None:
            self.train_writer.close()
        if self.valid_writer is not None:
            self.valid_writer.close()

    def test(self):
        # test the best validataion model
        test_oa, test_macc, test_cls_accs, test_cm = self.test_one_epoch(self.cfgs)
        self.print_cls_results(self.cfgs.start_epoch, test_oa, test_macc, test_cls_accs)
        if self.test_writer is not None:
            self.test_writer.add_scalar('test_oa', test_oa, self.cfgs.start_epoch)
            self.test_writer.add_scalar('test_macc', test_macc, self.cfgs.start_epoch)

        if self.test_writer is not None:
            self.test_writer.close()

    def train_one_epoch(self, epoch_idx, cfgs, dataloader):
        loss_meter = AverageMeter()
        confusion_matrix = ConfusionMatrix(n_class=cfgs.model.n_class)
        n_point = cfgs.model.n_point

        # self.scheduler.step()

        self.model.train()  # set model to train mode

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        num_iter = 0
        for batch_idx, (feature, coord, target) in pbar:
            num_iter += 1

            feature = feature.cuda(non_blocking=True)
            coord = coord.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            n_current_point = coord.shape[1]
            if n_current_point > n_point:  # point resampling strategy
                point_all = {1024: 1200, 2048: 2400, 4096: 4800, 8192: 8192}[n_point]
                if coord.size(1) < point_all:
                    point_all = coord.size(1)
                fps_idx = farthest_point_sample(coord, point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, n_point, False)]
                feature = index_points(feature, fps_idx).permute(0, 2, 1)
                coord = index_points(coord, fps_idx).permute(0, 2, 1)

            logits, rtkwargs = self.model(feature, coord)
            loss = self.criterion(logits, target.to(torch.int64), **rtkwargs)
            loss.backward()
            
            # optimize
            # if num_iter == cfg.step_per_update:
            if num_iter == 1:
                if cfgs.get('grad_norm_clip') is not None and cfgs.grad_norm_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfgs.grad_norm_clip, norm_type=2)
                num_iter = 0
                self.optimizer.step()
                self.model.zero_grad()

            # update confusion matrix
            confusion_matrix.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())
            # if batch_idx % cfg.print_freq == 0:
            if batch_idx % 1 == 0:
                pbar.set_description(f'Train Epoch [{epoch_idx}/{cfgs.max_epoch}] Loss: {loss_meter.val:.3f}, Overall Accuracy: {confusion_matrix.overall_accuray:.2f}')
        macc, oa, cls_accs = confusion_matrix.all_acc()
        return loss_meter.avg, oa, macc, cls_accs, confusion_matrix
    
    @torch.no_grad()
    def valid_one_epoch(self, cfgs, dataloader):
        return self.test_one_epoch(cfgs, dataloader)

    @torch.no_grad()
    def test_one_epoch(self, cfgs, dataloader):

        confusion_matrix = ConfusionMatrix(n_class=cfgs.model.n_class)
        n_point = cfgs.model.n_point

        self.model.eval()  # set model to eval mode

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (feature, coord, target) in pbar:

            feature = feature.cuda(non_blocking=True)[:, :n_point].permute(0, 2, 1)
            coord = coord.cuda(non_blocking=True)[:, :n_point].permute(0, 2, 1)
            target = target.cuda(non_blocking=True)

            logits, _ = self.model(feature, coord)

            confusion_matrix.update(logits.argmax(dim=1), target)
 
        macc, oa, cls_accs = confusion_matrix.all_acc()
        return oa, macc, cls_accs, confusion_matrix

if __name__ == '__main__':
    x = ManagerCls()