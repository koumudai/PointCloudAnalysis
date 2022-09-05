""" Scheduler Factory
Borrowed from Ross Wightman (https://www.github.com/timm)
"""
from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler


def build_scheduler_from_cfg(cfgs, optimizer, return_epochs=False):
    num_epochs = cfgs.n_epoch
    warmup_epochs = getattr(cfgs, 'warmup_epochs', 0)
    warmup_lr = getattr(cfgs, 'warmup_lr', 1.0e-6)  # linear warmup
    min_lr = cfgs.min_lr if getattr(cfgs, 'min_lr', False) else cfgs.lr/1000.
    cooldown_epochs = getattr(cfgs, 'cooldown_epochs', 0) 
    final_decay_rate = getattr(cfgs, 'final_decay_rate', 0.01)
    decay_rate = getattr(cfgs, 'decay_rate', None) or final_decay_rate**(1/num_epochs)
    decay_epochs = getattr(cfgs, 'decay_epochs', 1)
    t_max = getattr(cfgs, 't_max', num_epochs)
    if getattr(cfgs, 'lr_noise', None) is not None:
        lr_noise = getattr(cfgs, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(cfgs, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfgs, 'lr_noise_std', 1.),
        noise_seed=getattr(cfgs, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(cfgs, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(cfgs, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfgs, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if cfgs.name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_max,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=getattr(cfgs, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs
    elif cfgs.name == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs
    elif cfgs.name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **noise_args,
        )
    elif cfgs.name == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **noise_args,
        )
    elif cfgs.name == 'plateau':
        mode = 'min' if 'loss' in getattr(cfgs, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=cfgs.decay_rate,
            patience_t=cfgs.patience_epochs,
            lr_min=min_lr,
            mode=mode,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            cooldown_t=0,
            **noise_args,
        )
    elif cfgs.name == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=cfgs.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=getattr(cfgs, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs

    if return_epochs:
        return lr_scheduler, num_epochs
    else:
        return lr_scheduler