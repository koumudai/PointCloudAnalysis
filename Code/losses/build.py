from utils import registry


LOSSES = registry.Registry('loss')


def build_loss_from_cfg(cfg, default_args=None):
    return LOSSES.build(cfg, default_args=default_args)
