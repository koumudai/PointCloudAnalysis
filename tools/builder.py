import os
import sys
import torch
import torch.optim as optim
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler


def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    is_train = (config.others.subset == 'train')
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = is_train)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size = config.others.bs,
                                                 num_workers = int(args.num_workers),
                                                 drop_last = is_train,
                                                 worker_init_fn = worker_init_fn,
                                                 sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size = config.others.bs,
                                                 shuffle = is_train, 
                                                 drop_last = is_train,
                                                 num_workers = int(args.num_workers),
                                                 worker_init_fn=worker_init_fn)
    return sampler, dataloader


def model_builder(config):
    model = build_model_from_cfg(config)
    return model


def build_opti_sche(base_model, config):
    optimizer = build_optimizer(base_model, config)
    scheduler = build_scheduler(base_model, config, optimizer)
    return optimizer, scheduler


def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()
    return optimizer


def build_scheduler(base_model, config, optimizer):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler