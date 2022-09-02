import os
import yaml
import shutil
from easydict import EasyDict
from utils.logger import print_log


def log_args_to_file(args, pre='args', logger=None):
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
        try:
            ret = yaml.load(f, Loader=yaml.FullLoader)
        except:
            ret = yaml.load(f)
    return ret


def merge_cfgs(config, new_config):
    for key, val in new_config.items():
        if isinstance(val, dict):
            config[key] = merge_cfgs(EasyDict(), val)
        else:
            if key == '_base_':
                config[key] = merge_cfgs(EasyDict(), load_yaml(new_config['_base_']))
            else:
                config[key] = val
    return config


def get_cfgs_from_yaml(yaml_file):
    config = merge_cfgs(EasyDict(), load_yaml(yaml_file))
    return config


def get_cfgs(args, logger=None):
    if args.resume:
        cfg_file = f'{args.experiment_path}/config.yaml'
        if not os.path.exists(cfg_file):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_file}', logger = logger)
        args.cfg_file = cfg_file
    cfgs = get_cfgs_from_yaml(args.cfg_file)
    if not args.resume:
        save_experiment_config(args, logger)
    return cfgs


def save_experiment_config(args, logger=None):
    cfg_file = f'{args.experiment_path}/config.yaml'
    shutil.copy2(args.cfg_file, cfg_file)
    print_log(f'Copy the Config file from {args.cfg_file} to {cfg_file}', logger=logger)
