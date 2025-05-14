from typing import Type
import math
from functools import partial

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from accelerate import Accelerator

from src.utils.helpers import exists

# constants

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * 0.1)
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * (1 - 0.01) + 0.01
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1 - 0.01) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) + 0.01


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_train_steps: int, num_warmup_steps: int = None, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



# optimizer with scheduler

class OptimizerWithScheduler(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        max_grad_norm: float | None = 5.0,
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_kwargs)

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator
        self.total_norm = None

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if exists(self.max_grad_norm):
            # for param_group in self.optimizer.param_groups:
            all_params = [p for param_group in self.optimizer.param_groups for p in param_group['params']]
                
            total_norm = self.accelerator.clip_grad_norm_(all_params, self.max_grad_norm)
            # print(self.max_grad_norm)
            # print(total_norm)
            self.total_norm = total_norm

        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
