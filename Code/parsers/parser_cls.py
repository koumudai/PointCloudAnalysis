import os
import numpy as np
import torch
import argparse
from parsers.parser_utils import *
from pathlib import Path
from utils import misc


def get_args_and_cfgs():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='yaml config file')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu mode')
    parser.add_argument('--gpu_device', type=str, default='4, 5, 6, 7', help='specify gpu device')
    parser.add_argument('--num_workers', type=int, default=4)
    # seed
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--deterministic', action='store_true', default=False, help='whether to set deterministic options for CUDNN backend.')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--bs_train', type=int, default=120, help='train batch size')
    parser.add_argument('--bs_valid', type=int, default=120, help='valid batch size')
    parser.add_argument('--bs_test', type=int, default=120, help='test batch size')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normal')

    args = parser.parse_args()

    if args.cfg_file is None:
        raise ValueError('--cfg_file cannot be none')

    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both activate')

    cfg_name = Path(args.cfg_file).stem  # {model_name}_{dataset_name}_{n_point}pts
    if len(cfg_name.split('_')) != 3:
        raise ValueError(r'--cfg_file must conform to {model_name}_{dataset_name}_{n_point}pts.yaml')

    args.exp_name = ('test' if args.test else 'train') + f'_{args.exp_name}'
    
    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    args.use_gpu = args.use_gpu and torch.cuda.is_available()

    if args.seed < 0 or args.seed >= 10000:
        args.seed = np.random.randint(1, 10000)
    misc.set_random_seed(args.seed, args.deterministic)

    args.exp_path = f'./experiments/{"/".join(cfg_name.split("_"))}/{args.exp_name}'
    args.tfboard_path = f'./experiments/{"/".join(cfg_name.split("_"))}/TFBoard/{args.exp_name}'
    args.log_name = cfg_name

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
        print(f'Create experiment path successfully at {args.exp_path}')

    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print(f'Create TFBoard path successfully at {args.tfboard_path}')

    cfgs = get_cfgs(args)

    # batch size
    cfgs.dataset.train.others.bs = args.bs_train if args.bs_train != 0 else cfgs.bs_train
    cfgs.dataset.valid.others.bs = args.bs_valid if args.bs_valid != 0 else cfgs.bs_valid
    cfgs.dataset.test.others.bs = args.bs_test if args.bs_test != 0 else cfgs.bs_test
    # use_normals
    cfgs.model.use_normals = args.use_normals or cfgs.model.get('use_normals', False) # cfgs.model.use_normals is always equal to  args.use_normals
    cfgs.dataset.train.others.use_normals = args.use_normals
    cfgs.dataset.valid.others.use_normals = args.use_normals
    cfgs.dataset.test.others.use_normals = args.use_normals
    # lr
    cfgs.scheduler.lr = cfgs.optimizer.lr

    return args, cfgs

