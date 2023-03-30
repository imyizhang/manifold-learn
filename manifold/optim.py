import torch


def optimizer(
    params,
    lr: float,
    _optimizer: str = 'sgd',
    **kwargs,
) -> torch.optim.Optimizer:
    if _optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if _optimizer == 'adam':
        return torch.optim.Adam(params, lr=lr, **kwargs)
    raise ValueError(f"'{_optimizer}' optimizer is not supported")


def lr_scheduler(
    optimizer: torch.optim.Optimizer,
    _lr_scheduler: str = 'annealing',
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    if _lr_scheduler == 'annealing':
        return AnnealingLR(optimizer, **kwargs)
    raise ValueError(f"'{_lr_scheduler}' lr scheduler is not supported")


def annealing_lr(
    learning_rate,
    anneal_lr,
    lr_decay_rate,
    lr_min_factor,
    cur_epoch,
    total_epochs,
    decay_epochs=None,  # unused for now
    warmup_lr=0,
    warmup_epochs=0,
):
    """
    Decays the learning rate
    :param learning_rate: float Current learning rate
    :param anneal_lr: str Specifies the learning rate annealing. Must be one of "none", "linear" or "cosine"
    :param lr_decay_rate: float Rate of cosine decay.
    :param lr_min_factor: float Minimal learning rate of linear decay.
    :param cur_epoch: int Current epoch
    :param total_epochs: int Total number of epochs
    :param decay_epochs: int Number of decay epochs (unused)
    :param warmup_epochs: int Number of epochs for linearly warming up the learning rate
    :param warmup_lr: float Starting learning rate to warm up from.
    :return: float New learning rate
    """
    anneal_epochs = total_epochs - warmup_epochs
    if cur_epoch < warmup_epochs:
        lr = warmup_lr + (learning_rate - warmup_lr) * cur_epoch / warmup_epochs
    else:
        cur_epoch = cur_epoch - warmup_epochs
        if anneal_lr == "none":
            lr = learning_rate
        elif anneal_lr == "linear":
            lr = learning_rate * max(lr_min_factor, 1 - cur_epoch / anneal_epochs)
        elif anneal_lr == "cosine":
            eta_min = 0
            lr = (
                eta_min
                + (learning_rate - eta_min)
                * (1 + np.cos(np.pi * cur_epoch / anneal_epochs))
                / 2
            )
        else:
            raise RuntimeError(f"Unknown learning rate annealing “{anneal_lr = }”")

    return lr


class AnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
