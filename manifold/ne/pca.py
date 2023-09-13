from typing import Optional, Tuple

import torch

import manifold.transforms
from manifold.base import Estimator

# functional interface


def _flip_sign(
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


def svd(
    X: torch.Tensor,
    *,
    full_matrices: bool = False,
    driver: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies the singular value decomposition (SVD).

    References:
        [1] https://forums.fast.ai/t/svd-sign-ambiguity-for-pca-determinism/12480
    """
    # expect X to be a torch.half, torch.float or torch.double tensor
    # fixme: torch.linalg.svd is not implemented for torch.half
    half = False
    if X.dtype == torch.half:
        half = True
        X = X.float()
    U, S, Vh = torch.linalg.svd(
        X,
        full_matrices=full_matrices,
        driver=driver,
    )
    if half:
        U, S, Vh = U.half(), S.half(), Vh.half()
    return _flip_sign(U, S, Vh)


def pca(X: torch.Tensor, num_components: int) -> torch.Tensor:
    """Applies the principal component analysis (PCA).

    References:
        [1] https://forums.fast.ai/t/svd-sign-ambiguity-for-pca-determinism/12480
    """
    # expect X to be a torch.half, torch.float or torch.double tensor
    _, num_features = X.shape
    if num_components > num_features:
        raise ValueError(
            f"expected num_components must be a positive integer no greater than number of features, but got {num_components}"
        )
    X = manifold.transforms.center(X, dim=0)
    _, _, Vh = svd(X, full_matrices=False)
    # fixme: torch.matmul is not implemented for torch.half
    if X.dtype == torch.half:
        return torch.matmul(X.float(), Vh[:num_components].T.float()).half()
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
        _, _, Vh = svd(X - self.mean_, full_matrices=False)
        self.register_buffer("components_", Vh[: self.num_components])
        return self

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        # fixme: torch.matmul is not implemented for torch.half
        if X.dtype == torch.half:
            return torch.matmul(
                (X - self.mean_).float(), self.components_.T.float()
            ).half()
        return torch.matmul(X - self.mean_, self.components_.T)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit_transform()' instead"
        )

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "components_"):
            raise RuntimeError(f"call '{type(self).__name__}.fit()' first")
        # fixme: torch.matmul is not implemented for torch.half
        if Y.dtype == torch.half:
            return (
                torch.matmul(Y.float(), self.components_.float()).half()
                + self.mean_
            )
        return torch.matmul(Y, self.components_) + self.mean_
