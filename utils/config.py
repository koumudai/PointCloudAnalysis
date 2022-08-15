import yaml
from easydict import EasyDict
import os
from .logger import print_log


def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger=logger)


def log_cfgs_to_file(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger=logger)
            log_cfgs_to_file(cfgs[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger=logger)


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        try:
            ret = yaml.load(f, Loader=yaml.FullLoader)
        except:
            ret = yaml.load(f)
    return ret


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if isinstance(val, dict):
            config[key] = merge_new_config(EasyDict(), val)
        else:
            if key == '_base_':
                config[key] = merge_new_config(EasyDict(), load_yaml(new_config['_base_']))
            else:
                config[key] = val
    return config


def cfg_from_yaml_file(cfg_file):
    config = merge_new_config(EasyDict(), load_yaml(cfg_file))
    return config


def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger=logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger=logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config


def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger=logger )