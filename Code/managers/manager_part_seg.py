import os
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from managers import builder
from managers.manager_utils import ManagerBase
from models.model_utils import *
from utils.logger import *
from utils.metrics import *
from utils.misc import cal_model_parm_nums


class ManagerPartSeg(ManagerBase):
    def __init__(self, args, cfgs):
        super().__init__(args, cfgs)

    def build_transform(self):
        raise NotImplementedError()
    
    def build_other_info(self):
        cfg = self.cfgs.dataset.test if self.args.test else self.cfgs.dataset.train
        if self.args.class_choice is None:
            self.seg_num_all = sum(cfg.seg_num)
            self.seg_index_start = 0
        else:
            class_id = cfg.cat2id[self.args.class_choice]
            self.seg_num_all = cfg.seg_num[class_id]
            self.seg_index_start = cfg.index_start[class_id]

    def train(self):
        valid_metrics = MetricsShapeNetPart(0., 0., 0., [], [])
        best_epoch, best_valid_metrics = 0, MetricsShapeNetPart(0., 0., 0., [], [])
    
        start_epoch, end_epoch = self.cfgs.start_epoch, self.cfgs.max_epoch

        self.model.zero_grad()

        for epoch_idx in range(start_epoch + 1, end_epoch + 1):
            # train one epoch
            train_loss, train_metrics = self.train_one_epoch(epoch_idx, self.cfgs, self.train_dataloader)
            # valid one epoch
            valid_metrics = self.valid_one_epoch(self.cfgs, self.valid_dataloader)
            # save last checkpoint
            self.save_checkpoint('ckpt-last', epoch_idx)
            # save best checkpoint
            if valid_metrics.miou > best_valid_metrics.miou:
                best_epoch = epoch_idx
                best_valid_metrics.update(valid_metrics)
                self.logger.info(f'Find a better ckpt @Epoch {best_epoch}')
                self.print_results(best_epoch, best_valid_metrics)
                self.save_checkpoint('ckpt-best', best_epoch)
            
            self.logger.info(f'Epoch: {epoch_idx:3d}, train_miou: {train_metrics.miou:.2f}%, train_oa: {train_metrics.oa:.2f}%, train_macc: {train_metrics.macc:.2f}%, \n'
                f'{" "*12}valid_miou: {valid_metrics.miou:.2f}%, valid_oa: {valid_metrics.oa:.2f}%, valid_macc: {valid_metrics.macc:.2f}%, \n'
                f'{" "*12}best_valid_miou: {best_valid_metrics.miou:.2f}%, valid_oa_when_best_valid_miou: {best_valid_metrics.oa:.2f}%, valid_macc_when_best_valid_miou: {best_valid_metrics.macc:.2f}%.')

            if self.train_writer is not None:
                self.train_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.train_writer.add_scalar('train_loss', train_loss, epoch_idx)
                self.train_writer.add_scalar('train_miou', train_metrics.miou, epoch_idx)
                self.train_writer.add_scalar('train_oa', train_metrics.oa, epoch_idx)
                self.train_writer.add_scalar('train_macc', train_metrics.macc, epoch_idx)
                # TODO: add cls_ious and cls_accs

            if self.valid_writer is not None:
                self.valid_writer.add_scalar('epoch', epoch_idx, epoch_idx)
                self.valid_writer.add_scalar('valid_miou', valid_metrics.miou, epoch_idx)
                self.valid_writer.add_scalar('valid_oa', valid_metrics.oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc', valid_metrics.macc, epoch_idx)
                # TODO: add cls_ious and cls_accs

                self.valid_writer.add_scalar('best_epoch', best_epoch, epoch_idx)
                self.valid_writer.add_scalar('best_valid_miou', best_valid_metrics.miou, epoch_idx)
                self.valid_writer.add_scalar('valid_oa_when_best_valid_miou', best_valid_metrics.oa, epoch_idx)
                self.valid_writer.add_scalar('valid_macc_when_best_valid_miou', best_valid_metrics.macc, epoch_idx)
                # TODO: add cls_ious and cls_accs

        if self.train_writer is not None:
            self.train_writer.close()
        if self.valid_writer is not None:
            self.valid_writer.close()

    def test(self):
        # test the best validataion model
        test_metrics = self.test_one_epoch(self.cfgs)
        self.print_results(self.cfgs.start_epoch, test_metrics)
        if self.test_writer is not None:
            self.valid_writer.add_scalar('test_miou', test_metrics.miou, self.cfgs.start_epoch)
            self.valid_writer.add_scalar('test_oa', test_metrics.oa, self.cfgs.start_epoch)
            self.valid_writer.add_scalar('test_macc', test_metrics.macc, self.cfgs.start_epoch)
                # TODO: add cls_ious and cls_accs

        if self.test_writer is not None:
            self.test_writer.close()

    def train_one_epoch(self, epoch_idx, cfgs, dataloader):
        loss_meter = AverageMeter()
        train_results = ResultShapeNetPart(self.args.class_choice, self.seg_num_all, self.seg_index_start)

        self.model.train()  # set model to train mode
        
        if self.scheduler is not None:
            self.scheduler.step(epoch_idx)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (points, label, seg_true) in pbar:
            seg_true -= self.seg_index_start
            label_one_hot = F.one_hot(label, num_classes=16).squeeze(1)
            points, label_one_hot, seg_true = points.cuda().permute(0, 2, 1), label_one_hot.cuda(), seg_true.cuda()

            seg_pred, rtkwargs = self.model(points, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = self.criterion(seg_pred.view(-1, self.seg_num_all), seg_true.view(-1), **rtkwargs)
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_norm_clip is not None and self.args.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip, norm_type=2)
            self.optimizer.step()

            loss_meter.update(loss.item())
            train_results.update(seg_true.view(-1).cpu().numpy(), seg_pred.view(-1).cpu().numpy(), seg_true.cpu().numpy(), seg_pred.cpu().numpy(), label.cpu().numpy())
            if batch_idx % self.args.print_freq == 0:
                pbar.set_description(f'Train Epoch [{epoch_idx}/{cfgs.max_epoch}] Loss: {loss_meter.val:.3f}, Mean IOU: {cm.miou:.2f}%, Overall Accuracy: {cm.overall_accuracy:.2f}%, Mean Accuracy: {cm.mean_accuracy:.2f}%')
        metrics = train_results.cal_results()
        return loss_meter.avg, metrics

    @torch.no_grad()
    def valid_one_epoch(self, cfgs, dataloader):
        return self.test_one_epoch(cfgs, dataloader)

    @torch.no_grad()
    def test_one_epoch(self, cfgs, dataloader):
        cm = ConfusionMatrixShapeNetPart(n_class=cfgs.model.n_seg)

        self.model.eval()  # set model to eval mode

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (points, label, seg_true) in pbar:

            seg_true -= self.seg_index_start
            label_one_hot = F.one_hot(label, num_classes=16)
            points, label_one_hot, seg_true = points.cuda().permute(0, 2, 1), label_one_hot.cuda(), seg_true.cuda()

            seg_pred, _ = self.model(points, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1)
            
            # update confusion matrix
            cm.update(seg_pred.argmax(dim=1), seg_true)
 
        metrics = cm.cal_metrics()
        return metrics

    def print_results(self, epoch, metrics):
        msg = f'Epoch: {epoch:4}\tOverall Accuracy: {oa:3.2f}%\tMean Accuracy: {macc:3.2f}%\n'
        msg = f'Classes Accuracy\n'
        for name, acc in zip(self.class_names, class_accs):
            msg += f'{name:10}: {acc:3.2f}%\n'
        self.logger.info(msg)


if __name__ == '__main__':
    x = ManagerPartSeg()