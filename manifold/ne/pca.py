from typing import Optional, Tuple

import torch

import manifold.metrics
from manifold.base import Estimator

# functional interface


def svd(
    X: torch.Tensor,
    *,
    full_matrices: bool = False,
    driver: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies the singular value decomposition (SVD)."""
    return torch.linalg.svd(
        X,
        full_matrices=full_matrices,
        driver=driver,
    )


def flip_sign(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flips the signs of (U, S, Vh) to ensure deterministic ouput from SVD.

    References:
        [1] https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py#L829
    """
    rows_max_abs = U.abs().argmax(dim=0)
    cols = torch.arange(U.size(dim=1), dtype=torch.int64, device=U.device)
    signs = torch.sign(U[rows_max_abs, cols])
    U *= signs
    Vh *= signs.view(-1, 1)
    return U, S, Vh


def pca(X: torch.Tensor, num_components: int) -> torch.Tensor:
    """Applies the principal component analysis (PCA).

    References:
        [1] https://forums.fast.ai/t/svd-sign-ambiguity-for-pca-determinism/12480
    """
    _, num_features = X.shape
    if num_components > num_features:
        raise ValueError(
            f"expected num_components mast be a positive integer no greater than num_features, but got {num_components}"
        )
    X = manifold.metrics.mean_normalize(X, dim=0)
    U, S, Vh = svd(X, full_matrices=False)
    U, S, Vh = flip_sign(U, S, Vh)
    return torch.matmul(X, Vh[:num_components].T)


# class interface


class PCA(Estimator):
    """Principal component analysis (PCA).

    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    def __init__(
        self,
        num_components: int = 2,
    ):
        super().__init__()
        self.num_components = num_components

    def forward(self):
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit()' instead"
        )

    def fit(self, X: torch.Tensor) -> "PCA":
        _, num_features = X.shape
        if self.num_components > num_features:
            raise ValueError(
                f"expected num_components mast be a positive integer no greater than num_features, but got {self.num_components}"
            )
        self.register_buffer("mean_", X.mean(dim=0))
        U, S, Vh = svd(X - self.mean_, full_matrices=False)
        U, S, Vh = flip_sign(U, S, Vh)
        self.register_buffer("components_", Vh[: self.num_components])
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return torch.matmul(X - self.mean_, self.components_.T)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit_transform()' instead"
        )

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "components_"):
            raise RuntimeError(f"call '{type(self).__name__}.fit()' first")
        return torch.matmul(Y, self.components_) + self.mean_
