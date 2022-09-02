from utils import registry


LOSSES = registry.Registry('loss')


def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion, defined by `loss_name`.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed criterion specified by `loss_name`.
    """
    return LOSSES.build(cfg, **kwargs)
