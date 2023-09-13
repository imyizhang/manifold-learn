from typing import Optional

import numpy
import torch

__all__ = (
    "to_tensor",
    "transform",
)


# functional interface


def to_tensor(
    data: numpy.ndarray,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Converts a `numpy.ndarray` to a `torch.Tensor`."""
    return torch.tensor(data, dtype=dtype, device=device)


def center(
    X: torch.Tensor,
    *,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Centers X, or each feature of X, to have zero mean.

    Shapes:
        - X: :math:`(*, D)`
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    Xmean = X.mean() if dim is None else X.mean(dim=dim, keepdim=True)
    return X - Xmean


def minmax_normalize(
    X: torch.Tensor,
    *,
    dim: Optional[int] = None,
    min: float = 0.0,
    max: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scales X, or each feature of X, to a given range of [min, max].

    Shapes:
        - X: :math:`(*, D)`
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    Xmin = X.min() if dim is None else X.min(dim=dim, keepdim=True).values
    Xmax = X.max() if dim is None else X.max(dim=dim, keepdim=True).values
    # fixme: torch.clamp is not implemented for torch.half
    if X.dtype == torch.half:
        return min + (max - min) * (X - Xmin) / (Xmax - Xmin)
    return min + (max - min) * (X - Xmin) / (Xmax - Xmin).clamp(min=eps)


def mean_normalize(
    X: torch.Tensor,
    *,
    dim: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean normalizes X, or each feature of X.

    Shapes:
        - X: :math:`(*, D)`
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    Xmin = X.min() if dim is None else X.min(dim=dim, keepdim=True).values
    Xmax = X.max() if dim is None else X.max(dim=dim, keepdim=True).values
    # fixme: torch.clamp is not implemented for torch.half
    if X.dtype == torch.half:
        return center(X, dim=dim) / (Xmax - Xmin)
    return center(X, dim=dim) / (Xmax - Xmin).clamp(min=eps)


def standardize(
    X: torch.Tensor,
    *,
    dim: Optional[int] = None,
    correction: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standardizes X, or each feature of X, to have zero mean and unit variance.

    Shapes:
        - X: :math:`(*, D)`
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    Xstd = (
        X.std(correction=correction)
        if dim is None
        else X.std(dim=dim, correction=correction, keepdim=True)
    )
    # fixme: torch.clamp is not implemented for torch.half
    if X.dtype == torch.half:
        return center(X, dim=dim) / Xstd
    return center(X, dim=dim) / Xstd.clamp(min=eps)


def normalize(
    X: torch.Tensor,
    *,
    p: float = 2.0,
    dim: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Lp-normalizes Xor each sample of X.

    Shapes:
        - X: :math:`(*, D)`
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    # torch.nn.functional.normalize(x, p=p, dim=dim, eps=eps)
    Xnorm = X.norm(p=p) if dim is None else X.norm(p=p, dim=dim, keepdim=True)
    # fixme: torch.clamp is not implemented for torch.half
    if X.dtype == torch.half:
        return X / Xnorm
    return X / Xnorm.clamp(min=eps)


def transform(
    X: torch.Tensor,
    *,
    transformation: str,
    **kwargs,
) -> torch.Tensor:
    """Transforms X, or each feature of X.

    Shapes:
        - X: :math:`(*, D)`
    """
    if transformation == "center":
        return center(X, **kwargs)
    if transformation == "minmax_normalize":
        return minmax_normalize(X, **kwargs)
    if transformation == "mean_normalize":
        return mean_normalize(X, **kwargs)
    if transformation == "standardize":
        return standardize(X, **kwargs)
    if transformation == "normalize":
        return normalize(X, **kwargs)
    raise ValueError(f"transformation: '{transformation}' is not supported")


# class interface
