from typing import Optional

import torch

__all__ = (
    "distance",
    "pairwise_distance",
    "similarity",
    "pairwise_similarity",
    "negative_log_likelihood",
    "binary_cross_entropy",
    "cross_entropy",
    "kullback_leibler_divergence",
    "mle_loss",
    "nce_loss",
    "infonce_loss",
    "soft_nearest_neighbor_loss",
)


# functional interface


def cosine_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Cosine distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return 1.0 - cosine_similarity(x, y, dim=dim, keepdim=keepdim, eps=eps)


def correlation_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Pearson correlation distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return 1.0 - pearson_correlation_coefficient(
        x,
        y,
        dim=dim,
        keepdim=keepdim,
        eps=eps,
    )


def manhattan_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Measures the Manhattan distance between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # return (x - y).abs().sum(dim=dim, keepdim=keepdim)
    return (x - y).norm(p=1, dim=dim, keepdim=keepdim)


def squared_euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Measures the squared Euclidean distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return (x - y).square().sum(dim=dim, keepdim=keepdim)


def euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Measures the Euclidean distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # return (x - y).square().sum(dim=dim, keepdim=keepdim).sqrt()
    return (x - y).norm(p=2, dim=dim, keepdim=keepdim)


def minkowski_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    p: float = 2.0,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Measures the Minkowski distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # return ((x - y).abs() ** p).sum(dim=dim, keepdim=keepdim) ** (1.0 / p)
    return (x - y).norm(p=p, dim=dim, keepdim=keepdim)


def hamming_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Measures the Hamming distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return (x != y).sum(dim=dim, keepdim=keepdim)


def distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    metric: str = "euclidean",
    **kwargs,
) -> torch.Tensor:
    """Measures the distance between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    if metric == "cosine":
        return cosine_distance(x, y, **kwargs)
    if metric == "correlation":
        return correlation_distance(x, y, **kwargs)
    if metric == "manhattan":
        return manhattan_distance(x, y, **kwargs)
    if metric == "squared_euclidean":
        return squared_euclidean_distance(x, y, **kwargs)
    if metric == "euclidean":
        return euclidean_distance(x, y, **kwargs)
    if metric == "minkowski":
        return minkowski_distance(x, y, **kwargs)
    if metric == "hamming":
        return hamming_distance(x, y, **kwargs)
    raise ValueError(f"metric '{metric}' is not supported")


def pairwise_distance(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    metric: str = "euclidean",
    **kwargs,
) -> torch.Tensor:
    """Measures the pairwise distance between points within x or between x and y.

    References:
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # TODO
    raise NotImplementedError


def gaussian_kernel(
    distance: torch.Tensor,
    *,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (-distance / max(sigma**2, eps) / 2).exp()


def cauchy_kernel(
    distance: torch.Tensor,
    *,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 1 / (1 + distance / max(gamma**2, eps))


def inverse_kernel(
    distance: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 1 / distance.clamp(min=eps)


def negative_kernel(
    distance: torch.Tensor,
) -> torch.Tensor:
    return -distance


def euclidean_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Euclidean similarity between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return cauchy_kernel(
        squared_euclidean_distance(x, y, dim=dim, keepdim=keepdim),
        gamma=gamma,
        eps=eps,
    )


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Cosine similarity between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # return torch.nn.functional.cosine_similarity(x, y, dim=dim, eps=eps)
    dot_product = (x * y).sum(dim=dim, keepdim=keepdim)
    xnorm = x.norm(p=2, dim=dim, keepdim=keepdim)
    ynorm = y.norm(p=2, dim=dim, keepdim=keepdim)
    return dot_product / (xnorm * ynorm).clamp(min=eps)


def pearson_correlation_coefficient(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Pearson correlation coefficient between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    return cosine_similarity(
        x - x.mean(dim=dim, keepdim=keepdim),
        y - y.mean(dim=dim, keepdim=keepdim),
        dim=dim,
        keepdim=keepdim,
        eps=eps,
    )


def spearman_correlation_coefficient(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measures the Spearman correlation coefficient between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # TODO
    raise NotImplementedError


def similarity(
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    *,
    metric: str = "euclidean",
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Measures the similarity between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    if metric == "cosine":
        return cosine_similarity(x, y, **kwargs) / temperature
    if metric == "euclidean":
        return euclidean_similarity(x, y, **kwargs) / temperature
    if metric == "binary":
        return mask
    raise ValueError(f"metric '{metric}' is not supported")


def pairwise_similarity(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    metric: str = "euclidean",
    **kwargs,
) -> torch.Tensor:
    """Measures the pairwise similarity between points within x or between x and y.

    Shapes:
        - x: :math:`(*, D)`
        - y: :math:`(*, D)`
    """
    # TODO
    raise NotImplementedError


def log(
    input: torch.Tensor,
    clip_input: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Returns a new tensor with the natural logarithm of the elements of input."""
    if clip_input:
        return input.clamp(min=clip_min, max=clip_max).log()
    return input.log()


def negative_log_likelihood(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    with_log: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Measures the negative log likelihood loss between input probabilities and target.

    References:
        [1] https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

    Shapes:
        - input: :math:`(*, D)`
        - target: :math:`(*, D)`
    """
    if target.dtype == torch.bool:
        target = target.to(dtype=input.dtype)
    if with_log:
        loss = -target * log(input, **kwargs)
    else:
        loss = -target * input
    loss = loss.sum(dim=-1)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"reduction '{reduction}' is not supported")


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Mesures the binary cross entropy loss between input probabilities or logits and target.

    References:
        [1] https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        [2] https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    if target.dtype == torch.bool:
        target = target.to(dtype=input.dtype)
    # fixme: take advantage of the log-sum-exp trick for numerical stability
    return negative_log_likelihood(
        input,
        target,
        reduction=reduction,
        **kwargs,
    ) + negative_log_likelihood(
        1 - input,
        1 - target,
        reduction=reduction,
        **kwargs,
    )


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Measures the cross entropy loss between input probabilities or logits and target.

    References:
        [1] https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """
    if target.dtype == torch.bool:
        target = target.to(dtype=input.dtype)
    # fixme: take advantage of the log-sum-exp trick for numerical stability
    return negative_log_likelihood(input, target, reduction=reduction, **kwargs)


def kullback_leibler_divergence(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def mle_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    log_Z: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Measures the Maximum Likelihood Estimation (MLE) loss between similarities
    in high-dimensional space and similarities in low-dimensional space.
    """
    return negative_log_likelihood(Q, P, reduction=reduction, **kwargs) + log_Z


def nce_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    log_Z: torch.Tensor,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Measures the Noise Contrastive Estimation (NCE) loss between similarities
    in high-dimensional space and similarities in low-dimensional space.

    Given a batch of samples, the goal here is classify the samples as positive
    samples (data) or negative samples (noise).
    """
    posterior = Q / (Q + torch.exp(log_Z))
    return binary_cross_entropy(posterior, P, reduction=reduction, **kwargs)


def infonce_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    reduction: str = "mean",
    with_exp: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Measures the InfoNCE loss between similarities in high-dimensional space
    and similarities in low-dimensional space.

    Given a batch of samples, the goal here is to identify one positive sample
    (data) among the samples together with multiple negative samples (noise).
    """
    if with_exp:
        posterior = Q.softmax(dim=-1)
    else:
        posterior = Q / Q.sum(dim=-1, keepdim=True)
    return cross_entropy(posterior, P, reduction=reduction, **kwargs)


def soft_nearest_neighbor_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    reduction: str = "mean",
    with_log: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Measures the Soft Nearest Neighbor loss between similarities in
    high-dimensional space and similarities in low-dimensional space.

    Given a batch of samples, the goal here is to select a positive sample
    (data) among the samples together with multiple negative samples (noise).
    """
    posterior = (P * Q.softmax(dim=-1)).sum(dim=-1)
    if with_log:
        loss = -log(posterior, **kwargs)
    else:
        loss = -posterior
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"reduction '{reduction}' is not supported")


def criterion(name: str, **kwargs) -> "Metric":
    if name == "infonce":
        return InfoNCELoss(**kwargs)
    if name == "soft_nearest_neighbor":
        return SoftNearestNeighborLoss(**kwargs)
    raise ValueError(f"criterion '{name}' is not supported")


# class interface


Metric = torch.nn.Module


class Distance(Metric):
    def __init__(self, metric: str = "euclidean", **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return distance(x, y, metric=self.metric, **self.kwargs)


class PairwiseDistance(Metric):
    def __init__(self, metric: str = "euclidean", **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pairwise_distance(x, y, metric=self.metric, **self.kwargs)


class Similarity(Metric):
    def __init__(self, metric: str = "euclidean", **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return similarity(x, y, metric=self.metric, **self.kwargs)


class PairwiseSimilarity(Metric):
    def __init__(self, metric: str = "euclidean", **kwargs) -> None:
        super().__init__()
        self.metric = metric
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return pairwise_similarity(x, y, metric=self.metric, **self.kwargs)


class NegativeLogLikelihoodLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return negative_log_likelihood(
            input, target, reduction=self.reduction, **self.kwargs
        )


class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return binary_cross_entropy(
            input, target, reduction=self.reduction, **self.kwargs
        )


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return cross_entropy(
            input, target, reduction=self.reduction, **self.kwargs
        )


class MLELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        log_Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return mle_loss(P, Q, log_Z, reduction=self.reduction, **self.kwargs)


class NCELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        log_Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return nce_loss(P, Q, log_Z, reduction=self.reduction, **self.kwargs)


class InfoNCELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        return infonce_loss(P, Q, reduction=self.reduction, **self.kwargs)


class SoftNearestNeighborLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        return soft_nearest_neighbor_loss(
            P, Q, reduction=self.reduction, **self.kwargs
        )
