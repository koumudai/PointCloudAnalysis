import _init_path
from utils.logger import *
from parsers.parser_utils import *
import parsers.parser_cls as parser
from managers.manager_cls import ManagerCls
import time


def main():
    # args and cfgs
    args, cfgs = parser.get_args_and_cfgs()
    # logger
    logger = get_root_logger(log_file=f'{args.exp_path}/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.log', name=args.log_name)
    # log 
    log_args_to_file(args, 'args', logger=logger)
    log_cfgs_to_file(cfgs, 'cfgs', logger=logger)
    # manager
    manager = ManagerCls(args, cfgs)
    # train or test
    if not args.test:
        manager.train()
    else:
        manager.test()


if __name__ == '__main__':
    main()
