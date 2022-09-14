import _init_path
from utils.logger import *
from parsers import *
from managers import *
import time


def main():
    # args and cfgs
    args, cfgs = ParserPartSeg.get_args_and_cfgs()
    # logger
    logger = get_root_logger(log_file=f'{args.exp_path}/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.log', name=args.log_name)
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_cfgs_to_file(cfgs, 'cfgs', logger=logger)
    # manager
    manager = ManagerPartSeg(args, cfgs)
    # train or test
    if not args.test:
        manager.train()
    else:
        manager.test()


if __name__ == '__main__':
    main()
