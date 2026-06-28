"""Custom cosine annealing scheduler with linear warmup."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine annealing schedule with linear warmup.

    During the warmup phase (step < warmup_steps), the learning rate increases
    linearly from 0 to the base learning rate.  After warmup the learning rate
    follows a cosine curve that decays from the base learning rate down to
    ``base_lr * min_lr_ratio``.

    Parameters
    ----------
    optimizer:
        Wrapped optimizer whose learning rate will be scheduled.
    warmup_steps:
        Number of steps over which to linearly ramp the learning rate.
    total_steps:
        Total number of training steps (including warmup).
    min_lr_ratio:
        Minimum learning rate expressed as a fraction of the base learning
        rate.  Defaults to ``0.1``.

    Returns
    -------
    LambdaLR
        A PyTorch lambda learning-rate scheduler.
    """

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
