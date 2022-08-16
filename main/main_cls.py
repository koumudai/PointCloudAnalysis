import _init_path
from tools import train
# from tools import pretrain_run_net as pretrain_net
# from tools import finetune_run_net as finetune_net
# from tools import test_run_net as test_net
from utils import dist_utils
from utils import misc
from utils.logger import *
from parsers.parser_utils import *
import parsers.parser_cls as parser
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
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
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
    if args.distributed:
        assert cfgs.bs_train % world_size == 0 and cfgs.bs_valid % world_size == 0 and cfgs.bs_test % world_size == 0
        cfgs.dataset.train.others.bs = cfgs.bs_train // world_size
        cfgs.dataset.valid.others.bs = cfgs.bs_valid // world_size
        cfgs.dataset.test.others.bs = cfgs.bs_test // world_size 
    else:
        cfgs.dataset.train.others.bs = cfgs.bs_train
        cfgs.dataset.valid.others.bs = cfgs.bs_valid
        cfgs.dataset.test.others.bs = cfgs.bs_test
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_cfgs_to_file(cfgs, 'cfgs', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    if args.shot != -1:
        cfgs.dataset.train.others.shot = args.shot
        cfgs.dataset.train.others.way = args.way
        cfgs.dataset.train.others.fold = args.fold
        cfgs.dataset.val.others.shot = args.shot
        cfgs.dataset.val.others.way = args.way
        cfgs.dataset.val.others.fold = args.fold
    
    # run
    train(args, cfgs)

    if args.test:
        test_net(args, cfgs)
    else:
        if args.finetune_model or args.scratch_model:
            finetune_net(args, cfgs, train_writer, valid_writer)
        else:
            pretrain_net(args, cfgs, train_writer, valid_writer)


if __name__ == '__main__':
    main()
