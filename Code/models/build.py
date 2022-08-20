from utils import registry


MODELS = registry.Registry('model')


def build_model_from_cfg(cfg, default_args=None):
    return MODELS.build(cfg, default_args=default_args)
