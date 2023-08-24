from typing import Optional, Tuple

import numpy
import torch

from manifold.base import Estimator

# functional interface

registry = {}


def register(func):
    name = func.__name__
    while name.startswith("_"):
        name = name[1:]
    if name in registry:
        raise ValueError(f"algorithm '{name}' is already registered")
    registry[name] = func
    return func


@register
def _knn(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """KNN (K-Nearest Neighbors).

    References:
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    try:
        import sklearn.neighbors
    except ImportError:
        raise RuntimeError(
            "'scikit-learn' is not installed, run `pip install scikit-learn` to install"
        )
    index = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric=metric,
        metric_params=metric_kwargs,
        n_jobs=n_jobs,
        **kwargs,
    )
    index.fit(X)
    if not include_self:
        X = None
    neighbor_distances, neighbor_indices = index.kneighbors(
        X=X,
        return_distance=True,
    )
    return neighbor_indices, neighbor_distances


@register
def _annoy(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Annoy (Approximate Nearest Neighbors Oh Yeah).

    References:
        [1] https://github.com/spotify/annoy
    """
    try:
        import annoy
    except ImportError:
        raise RuntimeError(
            "'Annoy' is not installed, run `pip install annoy` to install"
        )
    n_samples, n_features = X.shape
    n_trees = kwargs.get("n_trees", 20)
    index = annoy.AnnoyIndex(n_features, metric=metric)
    if random_state is not None:
        index.set_seed(random_state)
    [index.add_item(i, x) for i, x in enumerate(X)]
    if n_jobs is not None:
        index.build(n_trees=n_trees, n_jobs=n_jobs)
    else:
        index.build(n_trees=n_trees, n_jobs=-1)
    neighbor_distances = numpy.empty(
        (n_samples, n_neighbors), dtype=numpy.float64
    )
    neighbor_indices = numpy.empty((n_samples, n_neighbors), dtype=numpy.int64)
    if not include_self:
        n_neighbors += 1
    for i in range(n_samples):
        i_neighbor_indices, i_neighbor_distances = index.get_nns_by_item(
            i,
            n_neighbors,
            include_distances=True,
        )
        if not include_self:
            i_neighbor_indices = i_neighbor_indices[1:]
            i_neighbor_distances = i_neighbor_distances[1:]
        neighbor_indices[i, :] = i_neighbor_indices
        neighbor_distances[i, :] = i_neighbor_distances
    return neighbor_indices, neighbor_distances


@register
def _pynndescent(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """PyNNDescent for fast Approximate Nearest Neighbors.

    References:
        [1]  https://pynndescent.readthedocs.io/en/latest/
    """
    try:
        import pynndescent
    except ImportError:
        raise RuntimeError(
            "'PyNNDescent' is not installed, run `pip install pynndescent` to install"
        )
    n_samples = X.shape[0]
    n_trees = min(64, 5 + round(n_samples**0.5 / 20.0))
    n_iters = max(5, round(numpy.log2(n_samples)))
    if not include_self:
        n_neighbors += 1
    index = pynndescent.NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwargs,
        n_trees=n_trees,
        n_iters=n_iters,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )
    neighbor_indices, neighbor_distances = index.neighbor_graph
    if not include_self:
        neighbor_indices = neighbor_indices[:, 1:]
        neighbor_distances = neighbor_distances[:, 1:]
    return neighbor_indices, neighbor_distances


@register
def _scann(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """ScaNN (Scalable Nearest Neighbors).

    References:
        [1] https://github.com/google-research/google-research/tree/master/scann
    """
    try:
        import scann
    except ImportError:
        raise RuntimeError(
            "'ScaNN' is not installed, run `pip install scann` to install"
        )
    if not include_self:
        n_neighbors += 1
    index = (
        scann.scann_ops_pybind.builder(
            X,
            n_neighbors,
            metric,
        )
        .tree(
            num_leaves=2000,
            num_leaves_to_search=100,
            training_sample_size=250000,
        )
        .score_ah(
            2,
            anisotropic_quantization_threshold=0.2,
        )
        .reorder(100)
        .build()
    )
    neighbor_indices, neighbor_distances = index.search_batched(
        n_neighbors,
        leaves_to_search=150,
    )
    if not include_self:
        neighbor_indices = neighbor_indices[:, 1:]
        neighbor_distances = neighbor_distances[:, 1:]
    return neighbor_indices, neighbor_distances


def _nearest_neighbors(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    algorithm: str = "annoy",
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # go to https://ann-benchmarks.com/ for more algorithms
    if algorithm not in registry:
        raise ValueError(f"algorithm '{algorithm}' is not supported")
    # force to return (numpy.int64, numpy.float64)
    return registry[algorithm](
        X,
        n_neighbors,
        metric=metric,
        metric_kwargs=metric_kwargs,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )


def nearest_neighbors(
    X: torch.Tensor,
    num_neighbors: int,
    *,
    algorithm: str = "annoy",
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Searches the nearest neighbors for each sample in X."""
    if generator is None:
        random_state = None
    else:
        seed = generator.initial_seed()
        # normally, seed should be between 0 and 2**32 - 1, i.e., uint32, but maximum value for seed is 2**31 - 1 in 'annoy'
        random_state = 0 if seed > 2**31 - 1 else seed
    # force using desired device
    device = X.device if device is None else device
    X = X.detach().cpu().numpy()
    neighbor_indices, neighbor_distances = _nearest_neighbors(
        X,
        num_neighbors,
        algorithm=algorithm,
        metric=metric,
        metric_kwargs=metric_kwargs,
        random_state=random_state,
        **kwargs,
    )
    neighbor_indices = torch.from_numpy(neighbor_indices).to(device=device)
    neighbor_distances = torch.from_numpy(neighbor_distances).to(device=device)
    return neighbor_indices, neighbor_distances


def to_neighbor_graph(
    nearest_neighbors: Tuple[torch.Tensor, torch.Tensor],
    /,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a graph given the nearest neighbors."""
    neighbor_indices, neighbor_distances = nearest_neighbors
    num_samples, num_neighbors = neighbor_indices.shape
    crow_indices = torch.arange(
        0,
        num_samples * num_neighbors + 1,
        num_neighbors,
        dtype=torch.int64,
    )
    col_indices = neighbor_indices.view(-1)
    neighbor_connectivities = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        torch.ones_like(col_indices),
    )
    neighbor_distances = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        neighbor_distances.view(-1),
    )
    return neighbor_connectivities, neighbor_distances


def neighbor_graph(
    X: torch.Tensor,
    num_neighbors: int,
    *,
    algorithm: str = "annoy",
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a graph of the nearest neighbors for each sample in X."""
    return to_neighbor_graph(
        nearest_neighbors(
            X,
            num_neighbors,
            algorithm=algorithm,
            metric=metric,
            metric_kwargs=metric_kwargs,
            generator=generator,
            device=device,
            **kwargs,
        )
    )


def pair(
    nearest_neighbors: Tuple[torch.Tensor, torch.Tensor],
    /,
) -> torch.Tensor:
    """Pairs each sample with its nearest neighbors."""
    neighbor_connectivities, _ = nearest_neighbors
    # the nearest neighbors or a graph for the nearest neighbors
    # https://github.com/pytorch/pytorch/issues/101385
    if not neighbor_connectivities.is_sparse_csr:
        # raise TypeError(
        #     f"expected 'torch.sparse_csr_tensor' for neighbor graph, but got '{type(neighbor_connectivities)}'"
        # )
        neighbor_connectivities, _ = to_neighbor_graph(nearest_neighbors)
    return neighbor_connectivities.to_sparse_coo().indices().T


def scale(
    nearest_neighbors: Tuple[torch.Tensor, torch.Tensor],
    /,
    start: int = 3,
    end: int = 6,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scales the nearest neighbors for each sample."""
    neighbor_indices, neighbor_distances = nearest_neighbors
    # compute scale factor
    sigma = neighbor_distances[:, start:end].mean(dim=1).clamp(min=eps)
    # rescale distances
    scaled_neighbor_distances = (
        neighbor_distances**2
        / sigma.unsqueeze(dim=1)
        / sigma[neighbor_indices]
    )
    # resort neighbors according to scaled distances
    scaled_neighbor_indices = neighbor_indices.gather(
        dim=1,
        index=scaled_neighbor_distances.argsort(dim=1),
    )
    return scaled_neighbor_indices, scaled_neighbor_distances


# class interface


class NearestNeighbors(Estimator):
    """Nearest neighbor searcher for each sample in X."""

    def __init__(
        self,
        num_neighbors: int,
        *,
        metric: str = "euclidean",
        metric_kwargs: dict = {},
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.generator = generator
        self.device = device
        self.kwargs = kwargs

    def forward(self):
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit()' instead"
        )

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> "NearestNeighbors":
        self.neighbor_indices, self.neighbor_distances = nearest_neighbors(
            X,
            self.num_neighbors,
            algorithm=self.algorithm,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            generator=self.generator,
            device=self.device,
            **self.kwargs,
        )
        return self

    def fit_transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(X, y=y)
        return self.neighbor_indices, self.neighbor_distances

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit_transform()' instead"
        )
