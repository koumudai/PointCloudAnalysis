import os
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn as nn

from managers import builder
from utils.logger import *
from utils.metrics import *
from models.pointcloud_utils import *

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

# from mmcv.utils.logging import get_logger
# logger = get_logger(config.experiment_name, osp.join(config.workdir, timestamp + '_log.txt'))
# logger.info()

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
        # build record file
        # self.build_record()
        # build tensorboard writer
        self.build_writer()
        # bulid transform
        # self.build_transform()
        # resume model
        # self.resume_model()
        # resume optimizer
        # self.resume_optimizer()
        # resume scheduler
        # self.resmue_scheduler()
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

        # load model parameters
        if self.args.test: # test
            builder.load_model(self.model, self.args.ckpts, logger=self.logger)
        else: # train
            # resume ckpts
            if self.args.resume:
                self.start_epoch, self.best_metric = builder.resume_model(self.model, self.args, logger=self.logger)

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

    def build_criterion(self):
        self.criterion = builder.build_criterion(self.cfgs.loss)

    def build_optimizer(self):
        self.optimizer = builder.build_optimizer(self.cfgs, self.model)

    def build_scheduler(self):
        self.scheduler = builder.build_scheduler(self.cfgs, self.model, self.optimizer)

    def build_record(self):
        raise NotImplementedError()

    def build_writer(self):
        if not self.args.test:
            self.train_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'train'))
            self.valid_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'valid'))
        else:
            self.test_writer = SummaryWriter(os.path.join(self.args.tfboard_path, 'test'))

    def build_transform(self):
        raise NotImplementedError()

    def resume_model(self):
        raise NotImplementedError()

    def resume_optimizer(self):
        raise NotImplementedError()

    def resume_scheduler(self):
        raise NotImplementedError()

    def train_net(self):
        valid_oa, valid_macc, valid_accs, valid_best_oa, valid_macc_when_best_oa, best_epoch = 0., 0., [], 0., 0., 0

        start_epoch, end_epoch = 0, self.cfgs.max_epoch

        self.model.zero_grad()

        for epoch_idx in range(start_epoch, end_epoch + 1):
            is_best = False

            # train one epoch
            train_loss, train_oa, train_macc, _, _ = self.train_one_epoch(epoch_idx, self.cfgs, self.train_dataloader)

            # valid one epoch
            # if epoch_idx % self.cfg.valid_freq == 0:
            if epoch_idx % 1 == 0:
                valid_oa, valid_macc, valid_accs, _ = self.valid_one_epoch(epoch_idx, self.cfgs, self.valid_dataloader)
                is_best = valid_oa > valid_best_oa
                if is_best:
                    valid_best_oa = valid_oa
                    valid_macc_when_best_oa = valid_macc
                    best_epoch = epoch_idx
                    print_log(f'Find a better ckpt @E{epoch_idx}', self.logger)
                    self.print_cls_results(valid_oa, valid_macc, valid_accs, epoch_idx, self.cfgs)

            
            lr = self.optimizer.param_groups[0]['lr']
            print_log(f'Epoch {epoch_idx} LR {lr:.6f} train_oa {train_oa:.2f}, valid_oa {valid_oa:.2f}, valid_best_oa {valid_best_oa:.2f}')
            if self.train_writer is not None:
                self.train_writer.add_scalar('lr', lr, epoch_idx)
                self.train_writer.add_scalar('train_loss', train_loss, epoch_idx)
                self.train_writer.add_scalar('train_oa', train_oa, epoch_idx)
                self.train_writer.add_scalar('train_macc', train_macc, epoch_idx)
                self.valid_writer.add_scalar('valid_oa', valid_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc', valid_macc, epoch_idx)
                self.valid_writer.add_scalar('valid_best_oa', valid_best_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc_when_best_oa', valid_macc_when_best_oa, epoch_idx)
                self.valid_writer.add_scalar('epoch', epoch_idx, epoch_idx)

            if is_best:
                metrics = {'train_oa': train_oa, 'train_macc': train_macc, 'valid_oa': valid_oa, 'valid_macc': valid_macc}
                self.save_checkpoint(epoch_idx, prefix='ckpt-best')

        # test the last epoch
        test_macc, test_oa, test_accs, test_cm = test_one_epoch(model, test_loader, cfg)
        print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
        if writer is not None:
            writer.add_scalar('test_oa', test_oa, epoch)
            writer.add_scalar('test_macc', test_macc, epoch)

        # test the best validataion model
        best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        test_macc, test_oa, test_accs, test_cm = test_one_epoch(model, test_loader, cfg)
        if writer is not None:
            writer.add_scalar('test_oa', test_oa, best_epoch)
            writer.add_scalar('test_macc', test_macc, best_epoch)
        print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

        if writer is not None:
            writer.close()

    def test_net(self):
        raise NotImplementedError()

    def train_one_epoch(self, epoch_idx, cfg, dataloader):
        loss_meter = AverageMeter()
        confusion_matrix = ConfusionMatrix(n_class=cfg.model.n_class)
        n_point = cfg.model.n_point

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
                if n_point == 1024:
                    point_all = 1200
                elif n_point == 2048:
                    point_all = 2400
                elif n_point == 4096:
                    point_all = 4800
                elif n_point == 8192:
                    point_all = 8192
                else:
                    raise NotImplementedError()
                if coord.size(1) < point_all:
                    point_all = coord.size(1)
                fps_idx = farthest_point_sample(coord, point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, n_point, False)]
                feature = index_points(feature, fps_idx).permute(0, 2, 1)
                coord = index_points(coord, fps_idx).permute(0, 2, 1)

            # print(feature.shape, coord.shape)
            logits, trans_feat = self.model(feature, coord)
            # print(logits.shape, trans_feat.shape)
            loss = self.criterion(logits, target.to(torch.int64), trans_feat)
            loss.backward()
            
            # optimize
            # if num_iter == cfg.step_per_update:
            if num_iter == 1:
                if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_norm_clip, norm_type=2)
                num_iter = 0
                self.optimizer.step()
                self.model.zero_grad()
                # if not cfg.sched_on_epoch:
                #     self.scheduler.step(epoch_idx)

            # update confusion matrix
            confusion_matrix.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())
            # if batch_idx % cfg.print_freq == 0:
            if batch_idx % 1 == 0:
                pbar.set_description(f'Train Epoch [{epoch_idx}/{cfg.max_epoch}] Loss {loss_meter.val:.3f} Acc {confusion_matrix.overall_accuray:.2f}')
        macc, overallacc, accs = confusion_matrix.all_acc()
        return loss_meter.avg, overallacc, macc, accs, confusion_matrix
    
    @torch.no_grad()
    def valid_one_epoch(self, epoch_idx, cfg, dataloader):

        confusion_matrix = ConfusionMatrix(n_class=cfg.model.n_class)
        n_point = cfg.model.n_point

        self.model.eval()  # set model to eval mode

        pbar = tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader))

        for batch_idx, (feature, coord, target) in pbar:

            feature = feature.cuda(non_blocking=True)
            coord = coord.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feature = feature[:, :n_point].permute(0, 2, 1)
            coord = coord[:, :n_point].permute(0, 2, 1)

            logits, trans_feat = self.model(feature, coord)

            confusion_matrix.update(logits.argmax(dim=1), target)
 
        tp, count = confusion_matrix.tp, confusion_matrix.count
        macc, overallacc, accs = confusion_matrix.cal_acc(tp, count)
        return overallacc, macc, accs, confusion_matrix

    @torch.no_grad()
    def test_one_epoch(self, epoch_idx):
        print('_test_each_epoch')
        pass

    def save_checkpoint(self, epoch_idx, prefix, metrics):
        torch.save({
                    'epoch': epoch_idx,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'metrics': metrics.state_dict(),
                    },
                    f'{self.args.experiment_path}/{prefix}.pth')
        print_log(f'Save checkpoint at {self.args.experiment_path}/{prefix}.pth', logger=self.logger)


    def print_cls_results(self, oa, macc, accs, epoch, cfg):
        s = f'\nClasses\tAcc\n'
        for name, acc_tmp in zip(self.class_names, accs):
            s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
        s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
        print_log(s, self.logger)

if __name__ == '__main__':
    x = ManagerCls()