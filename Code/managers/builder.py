import os
import numpy as np
import torch
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from losses import build_criterion_from_cfg
from timm.optim.optim_factory import create_optimizer
from timm.scheduler.scheduler_factory import create_scheduler


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_class_names(cfgs):
    return cfgs.cat2id.keys()


def build_dataset(args, cfgs):
    dataset = build_dataset_from_cfg(cfgs)
    is_train = not (cfgs.subset == 'test')
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size = cfgs.bs,
                                             shuffle = is_train, 
                                             drop_last = is_train and (len(dataset) >= 100),
                                             num_workers = int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    return dataloader


def build_model(cfgs):
    model = build_model_from_cfg(cfgs)
    return model


def build_criterion(cfgs):
    criterion = build_criterion_from_cfg(cfgs)
    return criterion


def build_optimizer(cfgs, model):
    optimizer = create_optimizer(cfgs, model)
    return optimizer


def build_scheduler(cfgs, optimizer):
    scheduler, _ = create_scheduler(cfgs, optimizer)
    return scheduler


def build_transform(cfgs):
    raise NotImplementedError()


def save_checkpoint(ckpt_path, epoch_idx, model, optimizer, scheduler, logger):
    torch.save({
        'epoch': epoch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        }, ckpt_path)
    logger.info(f'[Checkpoint] Save checkpoint at {ckpt_path}')


def load_checkpoint(ckpt_path, model, logger):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f'[Checkpoint] No checkpoint file from path {ckpt_path}...')
        
    # load state dict
    logger.info(f'[Load] loading checkpoint {ckpt_path}')
    state_dict = torch.load(ckpt_path)

    epoch_idx = state_dict['epoch']
    model.load_state_dict(state_dict['model'])

    return epoch_idx

def resume_checkpoint(ckpt_path, model, optimizer, scheduler, logger):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError(f'[Checkpoint] No checkpoint file from path {ckpt_path}...')

    # load state dict
    logger.info(f'[Resume] loading checkpoint {ckpt_path}')
    ckpt = torch.load(ckpt_path)

    epoch_idx = ckpt['epoch']
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

    return epoch_idx