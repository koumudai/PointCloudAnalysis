import numpy as np
import torch
from typing import List
import numpy as np
import sklearn.metrics as metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrixBase:
    """Accumulate a confusion matrix for a classification task."""
    def __init__(self, n_class, ignore_idx=None):
        self.value = 0
        self.n_class = n_class
        self.n_virtual_class = n_class if ignore_idx is None else n_class + 1
        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def update(self, pred, true): 
        """Update the confusion matrix with the given predictions."""
        true = true.flatten()
        pred = pred.flatten()
        if self.ignore_idx is not None:
            if (true == self.ignore_idx).sum() > 0:
                pred[true == self.ignore_idx] = self.n_virtual_class -1 
                true[true == self.ignore_idx] = self.n_virtual_class -1  
        unique_mapping = true.flatten() * self.n_virtual_class + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.n_virtual_class**2)
        self.value += bins.view(self.n_virtual_class, self.n_virtual_class)[:self.n_class, :self.n_class]

    def reset(self):
        """Reset all accumulated values."""
        self.value = 0

    @property
    def tp(self):
        """Get the true positive samples per-class."""
        return self.value.diag()
    
    @property
    def actual(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=1)

    @property
    def predicted(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=0)
    
    @property
    def fn(self):
        """Get the false negative samples per-class."""
        return self.actual - self.tp

    @property
    def fp(self):
        """Get the false positive samples per-class."""
        return self.predicted - self.tp

    @property
    def tn(self):
        """Get the true negative samples per-class."""
        actual = self.actual
        predicted = self.predicted
        return actual.sum() + self.tp - (actual + predicted)

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        # return self.tp + self.fn
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denomenator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        return self.value.sum()

    @property
    def overall_accuracy(self):
        return self.tp.sum() / self.total * 100

    @property
    def mean_accuracy(self):
        cls_accs = self.tp / self.count.clamp(min=1)
        return torch.mean(cls_accs) * 100

    @property
    def union(self):
        return self.value.sum(dim=0) + self.value.sum(dim=1) - self.value.diag()

    @staticmethod
    def cal_acc(tp, count):
        cls_accs = tp / count.clamp(min=1)
        oa = tp.sum() / count.sum()
        macc = torch.mean(cls_accs)  # class accuracy
        return oa.item() * 100, macc.item() * 100, cls_accs.cpu().numpy() * 100


class ConfusionMatrixShapeNetPart:
    def __init__(self, n_cls, n_seg, seg_num_list, index_start):
        self.n_cls = n_cls
        self.n_seg = n_seg
        self.seg_num_list = seg_num_list
        self.index_start = index_start
        self.cls_matrix = np.zeros((n_cls, n_cls))
        self.seg_matrix = [np.zeros((e, e)) for e in seg_num_list] if n_cls != 1 else np.zeros((n_seg, n_seg))

    def __init__(self, n_class, ignore_idx=None):
        self.value = 0
        self.n_class = n_class
        self.n_virtual_class = n_class if ignore_idx is None else n_class + 1
        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def update(self, pred, true): 
        """Update the confusion matrix with the given predictions."""
        true = true.flatten()
        pred = pred.flatten()
        if self.ignore_idx is not None:
            if (true == self.ignore_idx).sum() > 0:
                pred[true == self.ignore_idx] = self.n_virtual_class -1 
                true[true == self.ignore_idx] = self.n_virtual_class -1  
        unique_mapping = true.flatten() * self.n_virtual_class + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.n_virtual_class**2)
        self.value += bins.view(self.n_virtual_class, self.n_virtual_class)[:self.n_class, :self.n_class]

    def cal_metrics(self):
        tp, fp, fn = self.tp, self.fp, self.fn
  
        cls_ious = (tp / (tp + fp + fn).clamp(min=1)).cpu().numpy()
        cls_accs = (tp / self.count.clamp(min=1)).cpu().numpy()
        oa = tp.sum() / self.total

        miou = np.mean(cls_ious)
        macc = np.mean(cls_accs)  # class accuracy
        return MetricsShapeNetPart(miou.item() * 100, oa.item() * 100, macc.item() * 100, cls_ious * 100, cls_accs * 100)


class MetricsShapeNetPart:
    def __init__(self, miou=0., oa=0., macc=0., cls_ious=[], cls_accs=[]):
        self.miou = miou
        self.oa = oa
        self.macc = macc
        self.cls_ious = cls_ious
        self.cls_accs = cls_accs

    def update(self, other):
        self.miou = other.miou
        self.oa = other.oa
        self.macc = other.macc
        self.cls_ious = other.cls_ious
        self.cls_accs = other.cls_accs


def cal_iou(seg_true_all, seg_pred_all, seg_label_all, class_choice, seg_num_list, index_start_list, eva=False):
    shape_ious, category = [], {}
    for seg_pred, seg_true, seg_label in zip(seg_true_all, seg_pred_all, seg_label_all):
        if not class_choice:
            index_start = index_start_list[seg_label]
            seg_num = seg_num_list[seg_label]
            parts = range(index_start, index_start + seg_num)
        else:
            parts = range(seg_num_list[seg_label])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(seg_pred == part, seg_true == part))
            U = np.sum(np.logical_or(seg_pred == part, seg_true == part))
            iou = 1 if U == 0 else float(I) / U # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        category[seg_label].append(shape_ious[-1]) if seg_label in category else [shape_ious[-1]]

    return shape_ious, category if eva else shape_ious


class ResultShapeNetPart:
    def __init__(self, class_choice, seg_num_list, index_start_list):
        self.cls_true_all = []
        self.cls_pred_all = []
        self.seg_true_all = []
        self.seg_pred_all = []
        self.seg_label_all = []
        self.class_choice = class_choice
        self.seg_num_list = seg_num_list
        self.index_start_list = index_start_list

    def update(self, cls_true, cls_pred, seg_true, seg_pred, seg_label):
        assert len(cls_true.shape()) == len(cls_true.shape()) == len(seg_label.shape()) == 1
        assert len(seg_pred.shape()) == len(seg_true.shape()) == 2
        self.cls_true_all.append(cls_true)
        self.cls_pred_all.append(cls_pred)
        self.seg_true_all.append(seg_true)
        self.seg_pred_all.append(seg_pred)
        self.seg_label_all.append(seg_label)
    
    def cal_results(self, eva=False):
        assert not eva
        cls_true_all = np.concatenate(self.cls_true_all)
        cls_pred_all = np.concatenate(self.cls_pred_all)
        seg_true_all = np.concatenate(self.seg_true_all, axis=0)
        seg_pred_all = np.concatenate(self.seg_pred_all, axis=0)
        seg_label_all = np.concatenate(self.seg_label_all)
        oa = metrics.accuracy_score(cls_true_all, cls_pred_all)
        macc = metrics.balanced_accuracy_score(cls_true_all, cls_pred_all)
        miou = self.cal_iou(seg_true_all, seg_pred_all, seg_label_all, self.class_choice, self.seg_num_list, self.index_start_list)
        metrics = MetricsShapeNetPart(miou, oa, macc)
        return metrics