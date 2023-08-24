import math
import warnings
from typing import Optional, Sequence, Union

import torch

# functional interface


def optimizer(
    optimizer_: str,
    /,
    params: list,
    lr: float,
    **kwargs,
) -> torch.optim.Optimizer:
    """Maps an optimizer name to a `torch.optim.Optimizer`.

    References:
        [1] https://pytorch.org/docs/stable/optim.html#algorithms
    """
    if optimizer_ == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if optimizer_ == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    raise ValueError(f"optimizer '{optimizer_}' is not supported")


def lr_scheduler(
    lr_scheduler_: str,
    /,
    optimizer: torch.optim.Optimizer,
    annealing: str,
    **kwargs,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Maps an learning rate scheduler name to a `torch.optim.lr_scheduler.LRScheduler`.

    References:
        [1] https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    if lr_scheduler_ == "warmup":
        return AnnealingWithWarmup(optimizer, annealing, **kwargs)
    if lr_scheduler_ == "warm_restarts":
        return AnnealingWithWarmRestarts(optimizer, annealing, **kwargs)
    raise ValueError(
        f"learning rate scheduler '{lr_scheduler}' is not supported"
    )


def _annealing(
    annealing: str,
    /,
    T_curr: float,
    eta_max: float,
    T_max: int,
    eta_min: float = 0.0,
    gamma: Optional[float] = None,
) -> float:
    # remain constant
    if annealing == "none":
        return eta_max
    # linear annealing
    if annealing == "linear":
        return eta_min + (eta_max - eta_min) * (1 - T_curr / T_max)
    # cosine annealing
    if annealing == "cosine":
        return (
            eta_min
            + (eta_max - eta_min) * (1 + math.cos(math.pi * T_curr / T_max)) / 2
        )
    # exponential annealing
    if annealing == "exponential":
        if gamma <= 0 or gamma >= 1:
            raise ValueError(f"expected gamma between 0 and 1, but got {gamma}")
        return max(eta_min, eta_max * gamma**T_curr)
    raise ValueError(f"'{annealing}' annealing is not supported")


def annealing_with_warm_up(
    annealing: str,
    /,
    T_curr: float,
    eta_max: float,
    T_max: int,
    eta_min: float = 0.0,
    gamma: Optional[float] = None,
    T_warmup: int = 0,
    eta_warmup: Optional[float] = None,
):
    """Decays the learning rate after warmup steps."""
    if T_curr < T_warmup:
        if eta_warmup is None:
            eta_warmup = eta_max
        return eta_warmup + (eta_max - eta_warmup) * T_curr / T_warmup
    T_curr -= T_warmup
    T_max -= T_warmup
    return _annealing(
        annealing,
        T_curr,
        eta_max,
        T_max,
        eta_min,
        gamma=gamma,
    )


# class interface


class _enable_get_lr_call:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        self.obj._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.obj._get_lr_called_within_step = False
        return self


class AnnealingWithWarmup(torch.optim.lr_scheduler.LRScheduler):
    """Annealing learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        annealing: str,
        T_max: int,
        eta_min: float = 0.0,
        gamma: Optional[float] = None,
        T_warmup: int = 0,
        eta_warmup: Optional[float] = None,
        param_group: Optional[Union[int, Sequence[int]]] = None,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.annealing = annealing
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma
        self.T_warmup = T_warmup
        self.eta_warmup = eta_warmup
        self.T_curr = last_epoch
        if isinstance(param_group, int):
            param_group = [param_group]
        self.param_group = param_group
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "to get the last learning rate computed by the scheduler, please call method `get_last_lr()`"
            )
        return [
            annealing_with_warm_up(
                self.annealing,
                self.T_curr,
                base_lr,
                self.T_max,
                eta_min=self.eta_min,
                gamma=self.gamma,
                T_warmup=self.T_warmup,
                eta_warmup=self.eta_warmup,
            )
            for base_lr in self.base_lrs
        ]

    def step(
        self,
        epoch: Optional[float] = None,
    ) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_curr += 1
        else:
            if epoch < 0:
                raise ValueError(
                    f"expected epoch is non-negative, but got {epoch}"
                )
            self.T_curr = epoch
        self.last_epoch = math.floor(epoch)
        with _enable_get_lr_call(self):
            for i, data in enumerate(
                zip(
                    self.optimizer.param_groups,
                    self.get_lr(),
                )
            ):
                if self.param_group is not None and i not in self.param_group:
                    continue
                group, lr = data
                group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class AnnealingWithWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Annealing learning rate scheduler with warm restarts.

    References:
        [1] https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        annealing: str,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        gamma: Optional[float] = None,
        last_epoch: int = -1,
        param_group: Optional[Union[int, Sequence[int]]] = None,
        verbose: bool = False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                f"expected T_0 is a positive integer, but got {T_0}"
            )
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(
                f"expected T_mult is an integer no less than 1, but got {T_mult}"
            )
        self.annealing = annealing
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.gamma = gamma
        self.T_curr = last_epoch
        if isinstance(param_group, int):
            param_group = [param_group]
        self.param_group = param_group
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "to get the last learning rate computed by the scheduler, please call method `get_last_lr()`"
            )
        return [
            _annealing(
                self.annealing,
                self.T_curr,
                base_lr,
                self.T_i,
                eta_min=self.eta_min,
                gamma=self.gamma,
            )
            for base_lr in self.base_lrs
        ]

    def step(
        self,
        epoch: Optional[float] = None,
    ) -> None:
        if epoch is None and self.last_epoch == -1:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_curr += 1
            if self.T_curr >= self.T_i:
                self.T_curr -= self.T_i
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    f"expected epoch is non-negative, but got {epoch}"
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_i = self.T_0
                    self.T_curr = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1),
                            self.T_mult,
                        )
                    )
                    self.T_i = self.T_0 * self.T_mult**n
                    self.T_curr = epoch - (self.T_i - self.T_0) / (
                        self.T_mult - 1
                    )
            else:
                self.T_i = self.T_0
                self.T_curr = epoch
        self.last_epoch = math.floor(epoch)
        with _enable_get_lr_call(self):
            for i, data in enumerate(
                zip(
                    self.optimizer.param_groups,
                    self.get_lr(),
                )
            ):
                if self.param_group is not None and i not in self.param_group:
                    continue
                group, lr = data
                group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
