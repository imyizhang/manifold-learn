from typing import Optional
import math
import warnings

import torch

# functional interface


def optimizer(
    optimizer_: str,
    params,
    lr: float,
    **kwargs,
) -> torch.optim.Optimizer:
    if optimizer_ == 'sgd':
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if optimizer_ == 'adam':
        return torch.optim.Adam(params, lr=lr, **kwargs)
    raise ValueError(f"'{optimizer_}' optimizer is not supported")


def lr_scheduler(
    lr_scheduler_: str,
    optimizer: torch.optim.Optimizer,
    annealing: str,
    **kwargs,
) -> torch.optim.lr_scheduler.LRScheduler:
    if lr_scheduler_ == 'warmup':
        return AnnealingWithWarmup(optimizer, annealing, **kwargs)
    if lr_scheduler_ == 'warm_restarts':
        return AnnealingWithWarmRestarts(optimizer, annealing, **kwargs)
    raise ValueError(f"'{lr_scheduler}' lr scheduler is not supported")


def annealing_with_warm_restarts(
    annealing: str,
    T_curr: float,
    T_i: int,
    eta_min: float,
    eta_max: float,
    gamma: Optional[float] = None,
):
    if annealing == 'none':
        return eta_max
    if annealing == 'linear':
        return eta_min + (eta_max - eta_min) * (1 - T_curr / T_i)
    if annealing == 'exponential':
        return max(eta_min, eta_max * gamma**T_curr)
    if annealing == 'cosine':
        return eta_min + (eta_max -
                          eta_min) * (1 + math.cos(math.pi * T_curr / T_i)) / 2
    else:
        raise ValueError(f"'{annealing}' annealing is not supported")


def annealing_with_warm_up(
    annealing: str,
    T_curr: float,
    T_max: int,
    eta_min: float,
    eta_max: float,
    gamma: Optional[float] = None,
    T_warmup: int = 0,
    eta_warmup: Optional[float] = None,
):
    if T_curr < T_warmup:
        if eta_warmup is None:
            eta_warmup = eta_max
        return eta_warmup + (eta_max - eta_warmup) * T_curr / T_warmup
    T_curr -= T_warmup
    T_max -= T_warmup
    if annealing == 'none':
        return eta_max
    if annealing == 'linear':
        return eta_min + (eta_max - eta_min) * (1 - T_curr / T_max)
    if annealing == 'exponential':
        return max(eta_min, eta_max * gamma**T_curr)
    if annealing == 'cosine':
        return eta_min + (eta_max - eta_min) * (
            1 + math.cos(math.pi * T_curr / T_max)) / 2
    raise ValueError(f"'{annealing}' annealing is not supported")


# torch.nn.Module interface


class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False
        return self


class AnnealingWithWarmup(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        annealing: str,
        T_max: int,
        eta_min: float = 0,
        T_warmup: int = 0,
        eta_warmup: Optional[float] = None,
        last_epoch: int = -1,
        verbose=False,
    ) -> None:
        self.annealing = annealing
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_curr = last_epoch
        self.T_warmup = T_warmup
        self.eta_warmup = eta_warmup
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)

        return [
            annealing_with_warm_up(
                self.annealing,
                self.T_curr,
                self.T_max,
                self.eta_min,
                base_lr,
                T_warmup=self.T_warmup,
                eta_warmup=self.eta_warmup,
            ) for base_lr in self.base_lrs
        ]

    def step(
        self,
        epoch: Optional[float] = None,
        param_group: Optional[int] = None,
    ) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_curr += 1
        else:
            if epoch < 0:
                raise ValueError(
                    "expected non-negative epoch, but got {}".format(epoch))
            self.T_curr = epoch
        self.last_epoch = math.floor(epoch)
        with _enable_get_lr_call(self):
            for i, data in enumerate(
                    zip(
                        self.optimizer.param_groups,
                        self.get_lr(),
                    )):
                if (param_group is not None) and (i != param_group):
                    continue
                group, lr = data
                group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class AnnealingWithWarmRestarts(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        annealing: str,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"expected positive integer, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(
                f"expected integer no less than 1, but got {T_mult}")
        self.annealing = annealing
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_curr = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)
        return [
            annealing_with_warm_restarts(
                self.annealing,
                self.T_curr,
                self.T_i,
                self.eta_min,
                base_lr,
            ) for base_lr in self.base_lrs
        ]

    def step(
        self,
        epoch: Optional[float] = None,
        param_group: Optional[int] = None,
    ) -> None:

        if epoch is None and self.last_epoch == -1:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_curr += 1
            if self.T_curr >= self.T_i:
                self.T_curr = self.T_curr - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(
                    "expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_i = self.T_0
                    self.T_curr = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1),
                            self.T_mult,
                        ))
                    self.T_i = self.T_0 * self.T_mult**n
                    self.T_curr = epoch - (self.T_i - self.T_0) / (self.T_mult -
                                                                   1)
            else:
                self.T_i = self.T_0
                self.T_curr = epoch
        self.last_epoch = math.floor(epoch)
        with _enable_get_lr_call(self):
            for i, data in enumerate(
                    zip(
                        self.optimizer.param_groups,
                        self.get_lr(),
                    )):
                if (param_group is not None) and (i != param_group):
                    continue
                group, lr = data
                group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
