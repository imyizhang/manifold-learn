from typing import Optional

import torch

_pairwise_distances = {}


def register(pairwise_distances):
    metric = pairwise_distances.__name__
    if metric in _pairwise_distances:
        raise ValueError(f"'{metric}' is already registered")
    _pairwise_distances[metric] = pairwise_distances
    return pairwise_distances


@register
def manhattan(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return


@register
def euclidean(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return


@register
def cosine(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(X, Y, dim=-1, eps=eps)


@register
def hamming(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return


def pairwise_distances(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    *,
    metric: str = 'euclidean',
    **kwargs,
) -> torch.Tensor:
    if metric not in _pairwise_distances:
        raise ValueError(f"'{metric}' is not supported")
    return _pairwise_distances[metric](
        X,
        Y,
        **kwargs,
    )


def distance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    *,
    metric: str = 'euclidean',
    **kwargs,
) -> torch.Tensor:
    if metric not in _pairwise_distances:
        raise ValueError(f"'{metric}' is not supported")
    return _pairwise_distances[metric](
        X,
        Y,
        **kwargs,
    )


class PairwiseDistances(torch.nn.Module):

    def __init__(self, metric: str = 'euclidean', **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pairwise_distances(X, Y, self.metric, **self.kwargs)


def binary_cross_entropy(
    input,
    target,
    reducion: str = 'mean',
    min: float = 1e-4,
    max: float = 1.0,
):
    loss = -(input * torch.log(torch.clamp(target, min, max)) +
             (1 - input) * torch.log(torch.clamp(1 - target, min, max)))
    if reducion == 'mean':
        loss = loss.mean()
    return loss
