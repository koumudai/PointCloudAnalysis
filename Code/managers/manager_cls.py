import os
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn as nn
from managers import builder
from managers.manager_utils import ManagerBase
from models.model_utils import *
from utils.logger import *
from utils.metrics import *
from utils.misc import cal_model_parm_nums


class ManagerCls(ManagerBase):
    def __init__(self, args, cfgs):
        super().__init__(args, cfgs)

    def build_transform(self):
        raise NotImplementedError()

    def build_other_info(self):
        cfg = self.cfgs.dataset.test if self.args.test else self.cfgs.dataset.train
        self.class_names = cfg.cat2id.keys()

    def train(self):
        valid_oa, valid_macc, valid_accs, best_valid_oa, valid_macc_when_best_oa, best_epoch = 0., 0., [], 0., 0., 0

        start_epoch, end_epoch = self.cfgs.start_epoch, self.cfgs.max_epoch

        self.model.zero_grad()

        for epoch_idx in range(start_epoch + 1, end_epoch + 1):
            # train one epoch
            train_loss, train_oa, train_macc, _, _ = self.train_one_epoch(epoch_idx, self.cfgs, self.train_dataloader)
            # valid one epoch
            valid_oa, valid_macc, valid_accs, _ = self.valid_one_epoch(self.cfgs, self.valid_dataloader)
            # save last checkpoint
            self.save_checkpoint('ckpt-last', epoch_idx)
            # save best checkpoint
            if valid_oa > best_valid_oa:
                best_epoch = epoch_idx
                best_valid_oa = valid_oa
                valid_macc_when_best_oa = valid_macc
                self.logger.info(f'Find a better ckpt @Epoch {epoch_idx}')
                self.print_results(epoch_idx, valid_oa, valid_macc, valid_accs)
                self.save_checkpoint('ckpt-best', epoch_idx)
            
            self.logger.info(f'Epoch {epoch_idx}, train_oa: {train_oa:.2f}%, train_macc:{train_macc:.2f}%, valid_oa: {valid_oa:.2f}%, valid_macc: {valid_macc:.2f}%, best_valid_oa: {best_valid_oa:.2f}%, valid_macc_when_best_oa: {valid_macc_when_best_oa:.2f}%')
            if self.train_writer is not None:
                self.train_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.train_writer.add_scalar('train_loss', train_loss, epoch_idx)
                self.train_writer.add_scalar('train_oa', train_oa, epoch_idx)
                self.train_writer.add_scalar('train_macc', train_macc, epoch_idx)
            if self.valid_writer is not None:
                self.valid_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.valid_writer.add_scalar('valid_oa', valid_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc', valid_macc, epoch_idx)
                self.valid_writer.add_scalar('best_epoch', best_epoch, epoch_idx)
                self.valid_writer.add_scalar('best_valid_oa', best_valid_oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc_when_best_oa', valid_macc_when_best_oa, epoch_idx)

        if self.train_writer is not None:
            self.train_writer.close()
        if self.valid_writer is not None:
            self.valid_writer.close()

    def test(self):
        # test the best validataion model
        test_oa, test_macc, test_cls_accs, _ = self.test_one_epoch(self.cfgs)
        self.print_results(self.cfgs.start_epoch, test_oa, test_macc, test_cls_accs)
        if self.test_writer is not None:
            self.test_writer.add_scalar('test_oa', test_oa, self.cfgs.start_epoch)
            self.test_writer.add_scalar('test_macc', test_macc, self.cfgs.start_epoch)

        if self.test_writer is not None:
            self.test_writer.close()

    def train_one_epoch(self, epoch_idx, cfgs, dataloader):
        loss_meter = AverageMeter()
        cm = ConfusionMatrix(n_class=cfgs.model.n_class)

        self.model.train()  # set model to train mode
        
        if self.scheduler is not None:
            self.scheduler.step(epoch_idx)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (points, target) in pbar:

            points = points.cuda(non_blocking=True).permute(0, 2, 1)
            target = target.cuda(non_blocking=True).flatten().to(torch.int64)

            logits, rtkwargs = self.model(points)
            loss = self.criterion(logits, target, **rtkwargs)
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_norm_clip is not None and self.args.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip, norm_type=2)
            self.optimizer.step()

            # update confusion matrix
            cm.update(logits.argmax(dim=1), target)
            loss_meter.update(loss.item())
            if batch_idx % self.args.print_freq == 0:
                pbar.set_description(f'Train Epoch [{epoch_idx}/{cfgs.max_epoch}] Loss: {loss_meter.val:.3f}, Overall Accuracy: {cm.overall_accuracy:.2f}%, Mean Accuracy: {cm.mean_accuracy:.2f}%')

        oa, macc, cls_accs = cm.all_acc()
        return loss_meter.avg, oa, macc, cls_accs, cm

    @torch.no_grad()
    def valid_one_epoch(self, cfgs, dataloader):
        return self.test_one_epoch(cfgs, dataloader)

    @torch.no_grad()
    def test_one_epoch(self, cfgs, dataloader):
        cm = ConfusionMatrix(n_class=cfgs.model.n_class)

        self.model.eval()  # set model to eval mode

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (points, target) in pbar:

            points = points.cuda(non_blocking=True).permute(0, 2, 1)
            target = target.cuda(non_blocking=True)

            logits, _ = self.model(points)

            cm.update(logits.argmax(dim=1), target)
 
        oa, macc, cls_accs = cm.all_acc()
        return oa, macc, cls_accs, cm

    def print_results(self, epoch_idx, metrics):
        msg = f'Epoch: {epoch_idx:4}\tOverall Accuracy: {metrics.oa:3.2f}%\tMean Accuracy: {metrics.macc:3.2f}%\n'
        msg = f'Classes Accuracy\n'
        for name, acc in zip(self.class_names, metrics.class_accs):
            msg += f'{name:10}: {acc:3.2f}%\n'
        self.logger.info(msg)


if __name__ == '__main__':
    x = ManagerCls()