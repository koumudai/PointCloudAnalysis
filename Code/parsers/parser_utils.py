import os
import yaml
import shutil
from easydict import EasyDict
from utils.logger import print_log


def log_args_to_file(args, pre='args', logger=None):
    print(args)
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger=logger)


def log_cfgs_to_file(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            print_log(f'{pre}.{key} = EasyDict()', logger=logger)
            log_cfgs_to_file(cfgs[key], pre=f'{pre}.{key}', logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger=logger)


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def merge_cfgs(new_cfgs, old_cfgs):
    for k, v in old_cfgs.items():
        if isinstance(v, dict):
            new_cfgs[k] = merge_cfgs(EasyDict(), v)
        else:
            if k in new_cfgs.keys():
                assert new_cfgs[k] == v, f'the key "{k}" has different value'
            elif k == '_base_':
                cfgs = merge_cfgs(EasyDict(), load_yaml(old_cfgs['_base_']))
                for p, q in cfgs.items():
                    new_cfgs[p] = q
            else:
                new_cfgs[k] = v
    return new_cfgs
        

def get_cfgs_from_yaml(yaml_file):
    return merge_cfgs(EasyDict(), load_yaml(yaml_file))


def get_cfgs(args, logger=None):
    if args.resume:
        cfg_file = f'{args.exp_path}/config.yaml'
        if not os.path.exists(cfg_file):
            print_log("Failed to resume", logger=logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_file}', logger=logger)
        args.cfg_file = cfg_file
    cfgs = get_cfgs_from_yaml(args.cfg_file)
    if not args.resume:
        save_experiment_config(args, logger)
    return cfgs


def save_experiment_config(args, logger=None):
    cfg_file = f'{args.exp_path}/config.yaml'
    shutil.copy2(args.cfg_file, cfg_file)
    print_log(f'Copy the Config file from {args.cfg_file} to {cfg_file}', logger=logger)
