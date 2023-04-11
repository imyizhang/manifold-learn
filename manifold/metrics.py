from typing import Optional

import torch

# functional interface


def correlation_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 1. - pearson_correlation_coefficient(x, y, dim=dim, eps=eps)


def cosine_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 1. - cosine_similarity(x, y, dim=dim, eps=eps)


def manhattan_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return (x - y).abs().sum(dim=dim, keepdim=False)


def squared_euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return (x - y).square().sum(dim=dim, keepdim=False)


def euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return squared_euclidean_distance(x, y, dim=dim).sqrt()


def minkowski_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return


def hamming_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    return (x != y).sum(dim=dim, keepdim=False)


def gaussian_kernel(
    distance: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    distance /= sigma**2
    return torch.exp(-distance / 2)


def cauchy_kernel(
    distance: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    distance /= gamma**2
    return 1 / 1 + distance


def inverse_kernel(
    distance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 1 / (distance + eps)


def exp(
    input: torch.Tensor,
    scale_exp_input: bool = True,
    tau: float = 1.0,
) -> torch.Tensor:
    """Returns a new tensor with the exponential of the elements of input.
    """
    if scale_exp_input and (tau != 0):
        return torch.exp(input / tau)
    return torch.exp(input)


def log(
    input: torch.Tensor,
    clip_log_input: bool = True,
    clip_min: float = 1e-4,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Returns a new tensor with the natural logarithm of the elements of input.
    """
    if clip_log_input and (clip_min != 0):
        return torch.log(input.clip(min=clip_min, max=clip_max))
    return torch.log(input)


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        x,
        y,
        dim=dim,
        eps=eps,
    )


def pearson_correlation_coefficient(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        x - x.mean(dim=dim, keepdim=True),
        y - y.mean(dim=dim, keepdim=True),
        dim=dim,
        eps=eps,
    )


def spearman_corrcoef():
    return


def distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    metric: str = 'euclidean',
    **kwargs,
) -> torch.Tensor:
    return


def similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    metric: str = 'euclidean',
    **kwargs,
) -> torch.Tensor:
    return


def pairwise_distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    *,
    metric: str = 'euclidean',
    **kwargs,
) -> torch.Tensor:
    return


def negative_log_likelihood(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'sum',
    clip_log_input: bool = False,
    clip_min: float = 1e-4,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """The negative log likelihood loss.
    """
    if target.type == torch.bool:
        target = target.to(dtype=input.dtype)
    if target.shape == input.shape:
        loss = -target * log(
            input, clip_log_input, clip_min=clip_min, clip_max=clip_max)
        loss = loss.sum(dim=1)
        if reduction == 'sum':
            return loss.sum()
        if reduction == 'mean':
            return loss.mean()
        raise ValueError(f"'{reduction} 'reduction is not supported")
    return torch.nn.functional.nll_loss(
        log(input, clip_log_input, clip_min=clip_min, clip_max=clip_max),
        target,
        reduction=reduction,
    )


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'sum',
    with_logits: bool = False,
    clip_log_input: bool = False,
    clip_min: float = 1e-4,
    clip_max: float = 1.0,
) -> torch.Tensor:
    if target.dtype == torch.bool:
        target = target.to(dtype=input.dtype)
    if with_logits:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            reduction=reduction,
        )
    loss = -target * log(
        input,
        clip_log_input,
        clip_min=clip_min,
        clip_max=clip_max,
    ) - (1 - target) * log(
        1 - input,
        clip_log_input,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    loss = loss.sum(dim=1)
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    raise ValueError(f"'{reduction} 'reduction is not supported")


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'sum',
    with_logits: bool = False,
    clip_log_input: bool = False,
    clip_min: float = 1e-4,
    clip_max: float = 1.0,
) -> torch.Tensor:
    if target.type == torch.bool:
        target = target.to(dtype=input.dtype)
    if with_logits:
        return torch.nn.functional.cross_entropy(
            input,
            target,
            reduction=reduction,
        )
    return negative_log_likelihood(
        input,
        target,
        reduction=reduction,
        clip_log_input=clip_log_input,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def mle_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    reduction: str = 'sum',
) -> torch.Tensor:
    return negative_log_likelihood(Q, P, reduction=reduction)


def nce_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    log_Z: torch.Tensor,
    as_logistic_regression: bool = False,
    reduction: str = 'sum',
) -> torch.Tensor:
    if as_logistic_regression:
        # probability = torch.log(Q) - log_Z - torch.log(negative_samples)
        probability = torch.log(Q) - log_Z
        return binary_cross_entropy(
            probability,
            P,
            with_logits=True,
            reduction=reduction,
        )
    # posterior = Q / (Q + torch.exp(log_Z) * negative_samples)
    posterior = Q / (Q + torch.exp(log_Z))
    return binary_cross_entropy(posterior, P, reduction=reduction)


def infonce_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    log_Z: torch.Tensor,
    as_logistic_regression: bool = False,
    reduction: str = 'sum',
) -> torch.Tensor:
    if as_logistic_regression:
        probability = torch.log(Q) - log_Z
        return cross_entropy(
            probability,
            P,
            with_logits=True,
            reduction=reduction,
        )
    posterior = Q / Q.sum(dim=-1)
    return cross_entropy(posterior, P, reduction=reduction)


# torch.nn.Module interface


class Distance(torch.nn.Module):

    def __init__(self, metric: str = 'euclidean', **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pairwise_distance(X, Y, self.metric, **self.kwargs)


class PairwiseDistance(torch.nn.Module):

    def __init__(self, metric: str = 'euclidean', **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pairwise_distance(X, Y, self.metric, **self.kwargs)


class NLLLoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return negative_log_likelihood(
            input,
            target,
            reduction=self.reduction,
        )


class BCELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return binary_cross_entropy(
            input,
            target,
            reduction=self.reduction,
        )


class CELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return cross_entropy(
            input,
            target,
            reduction=self.reduction,
        )


class MLELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        log_Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return mle_loss(P, Q, reduction=self.reduction)


class NCELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        log_Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return nce_loss(P, Q, log_Z, reduction=self.reduction)


class InfoNCELoss(torch.nn.Module):

    def __init__(self, reduction: str = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        log_Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return infonce_loss(P, Q, log_Z, reduction=self.reduction)
