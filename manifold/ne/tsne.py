from typing import Optional

import torch

from .tscne import TSCNE, tscne

# functional interface


def tsne(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    num_components: int = 2,
    *,
    preprocess: bool = True,
    apply_pca: bool = True,
    max_features: int = 100,
    normalizing_dim: Optional[int] = None,
    num_neighbors: int = 10,
    algorithm: str = "annoy",
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    scale_nearest_neighbors: bool = True,
    scaling_start: int = 3,
    scaling_end: int = 6,
    num_positive_samples: int = 1,
    num_negative_samples: int = 1,
    force_resampling: bool = False,
    in_batch: bool = False,
    exclude_anchor_samples: bool = True,
    exclude_positive_samples: bool = True,
    exclude_neighbors: bool = True,
    replacement: bool = False,
    batch_size: Optional[int] = None,
    drop_last: bool = False,
    shuffle: bool = False,
    parametric: bool = False,
    init: str = "pca",
    encoder: str = "mlp",
    encoder_kwargs: dict = {},
    partition: float = 1.0,
    use_learnable_partition: bool = False,
    optimizer: str = "sgd",
    lr: float = 1.0,
    optimizer_kwargs: dict = {},
    annealing: str = "cosine",
    lr_scheduler: str = "warm_restarts",
    lr_scheduler_kwargs: dict = {
        # 'T_max': 10,
        "T_0": 50,
        "T_mult": 1,
        "eta_min": 0.1,
    },
    clip_grad: bool = True,
    clip_value: float = 4.0,
    num_epochs: int = 100,
    autosave: bool = False,
    root: str = "./",
    nearest_neighbors: Optional[str] = None,
    neighboring_samples: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    trained_model: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """t-distributed Stochastic Neighbor Embedding (t-SNE)."""
    return tscne(X, y=y, num_components=num_components)


# class interface


class TSNE(TSCNE):
    """t-distributed Stochastic Neighbor Embedding (t-SNE).

    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """

    def __init__(
        self,
        num_components: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_components = num_components
        self.kwargs = kwargs

    def forward(self):
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit()' instead"
        )

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> "TSNE":
        raise NotImplementedError
        # return self

    def fit_transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return tsne(
            X,
            y,
            num_components=self.num_components,
            **self.kwargs,
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit_transform()' instead"
        )

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
