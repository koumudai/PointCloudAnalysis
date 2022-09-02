import _init_path
from utils import dist_utils
from utils import misc
from utils.logger import *
from parsers.parser_utils import *
import parsers.parser_cls as parser
from managers.manager_cls import ManagerCls
import time
import os
import torch

def main():
    # args and cfgs
    args, cfgs = parser.get_args_and_cfgs()
    # CUDA
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # batch size
    cfgs.dataset.train.others.bs = args.bs_train if args.bs_train != 0 else cfgs.bs_train
    cfgs.dataset.valid.others.bs = args.bs_valid if args.bs_valid != 0 else cfgs.bs_valid
    cfgs.dataset.test.others.bs = args.bs_test if args.bs_test != 0 else cfgs.bs_test
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_cfgs_to_file(cfgs, 'cfgs', logger=logger)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed for augmentation
    # manager
    manager = ManagerCls(args, cfgs)

    if not args.test:
        manager.train_net()
    else:
        manager.test_net()

if __name__ == '__main__':
    main()
