import os
import argparse
from parsers.parser_utils import *
from pathlib import Path


def get_args_and_cfgs():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='yaml config file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_workers', type=int, default=4)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument( '--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=False, help='finetune modelnet with pretrained weight')
    parser.add_argument('--scratch_model', action='store_true', default=False, help='training modelnet from scratch')
    parser.add_argument('--label_smoothing', action='store_true', default=False, help='use label smoothing loss trick')
    parser.add_argument('--mode', choices=['easy', 'median', 'hard', None],default=None,help='difficulty mode for shapenet')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)

    args = parser.parse_args()

    if args.cfg_file is None:
        raise ValueError('--cfg_file cannot be none')

    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError('ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        raise ValueError('ckpts shouldnt be None while finetune_model mode')

    cfg_name = Path(args.cfg_file).stem  # {model_name}_{dataset_name}_{n_point}pts
    if len(cfg_name.split('_')) != 3:
        raise ValueError(r'--cfg_file must conform to {model_name}_{dataset_name}_{n_point}pts.yaml')

    args.exp_name = ('test' if args.test else 'train') + (f'_{args.exp_name}' if args.exp_name is not 'default' else '')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.mode is not None:
        args.exp_name = f'{args.exp_name}_{args.mode}'

    args.experiment_path = f'./experiments/{"/".join(cfg_name.split("_"))}/{args.exp_name}'
    args.tfboard_path = f'./experiments/{"/".join(cfg_name.split("_"))}/TFBoard/{args.exp_name}'
    args.log_name = cfg_name

    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

    cfgs = get_cfgs(args)

    return args, cfgs

