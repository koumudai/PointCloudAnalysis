import _init_path
from utils import dist_utils
from utils import misc
from utils.logger import *
from parsers.parser_utils import *
import parsers.parser_cls as parser
from tools.manager_cls import ManagerCls
import time
import os
import torch
from tensorboardX import SummaryWriter


def main():
    # args and cfgs
    args, cfgs = parser.get_args_and_cfgs()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    args.distributed = False
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            valid_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            valid_writer = None
    # batch size
    cfgs.dataset.train.others.bs = cfgs.bs_train
    cfgs.dataset.valid.others.bs = cfgs.bs_valid
    cfgs.dataset.test.others.bs = cfgs.bs_test
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_cfgs_to_file(cfgs, 'cfgs', logger=logger)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed for augmentation

    if args.shot != -1:
        cfgs.dataset.train.others.shot = args.shot
        cfgs.dataset.train.others.way = args.way
        cfgs.dataset.train.others.fold = args.fold
        cfgs.dataset.valid.others.shot = args.shot
        cfgs.dataset.valid.others.way = args.way
        cfgs.dataset.valid.others.fold = args.fold

    manager = ManagerCls(args, cfgs)

    if args.test:
        manager.test_net()
    else:
        # TODO: pretrain or finetune
        manager.train_net()


if __name__ == '__main__':
    main()
