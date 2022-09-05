import os
import numpy as np
import torch
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from losses import build_criterion_from_cfg
from optimizers import build_optimizer_from_cfg
from schedulers import build_scheduler_from_cfg


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataset(args, cfgs):
    dataset = build_dataset_from_cfg(cfgs._base_, cfgs.others)
    is_train = (cfgs.others.subset == 'train')

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size = cfgs.others.bs,
                                             shuffle = is_train, 
                                             drop_last = is_train,
                                             num_workers = int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    return dataloader, dataset.class_names


def build_model(cfgs):
    model = build_model_from_cfg(cfgs)
    return model


def build_criterion(cfgs):
    criterion = build_criterion_from_cfg(cfgs)
    return criterion


def build_optimizer(cfgs, model):
    optimizer = build_optimizer_from_cfg(model, cfgs.name, cfgs.lr, **cfgs.kwargs)
    return optimizer


def build_scheduler(cfgs, optimizer):
    scheduler = build_scheduler_from_cfg(cfgs, optimizer)
    return scheduler


def save_checkpoint(ckpt_path, epoch_idx, model, optimizer, scheduler, metrics, logger):
    torch.save({
        'epoch': epoch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'metrics': metrics,
        }, ckpt_path)
    logger.info(f'[Checkpoint] Save checkpoint at {ckpt_path}')


def load_checkpoint(ckpt_path, model, logger):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f'[Checkpoint] No checkpoint file from path {ckpt_path}...')
        
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    epoch_idx = state_dict['epoch']
    model.load_state_dict(state_dict['model'])

    return epoch_idx

def resume_checkpoint(ckpt_path, cfgs, model, optimizer, scheduler, logger):
    ckpt_path = ckpt_path if cfgs.ckpt_path is None else cfgs.ckpt_path
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f'[Checkpoint] No checkpoint file from path {ckpt_path}...')

    # load state dict
    logger.info(f'[Resume] loading checkpoint {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfgs.start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            logger.info('optimizer does not match')
    if scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except:
            logger.info('scheduler does not match')
    metrics = ckpt['metrics']

    return cfgs.start_epoch, metrics