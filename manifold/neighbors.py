from typing import Tuple, Optional, Union

import numpy as np
import torch

_nearest_neighbors = {}


def register(nearest_neighbors):
    algorithm = nearest_neighbors.__name__[1:]
    if algorithm in _nearest_neighbors:
        raise ValueError(f"'{algorithm}' is already registered")
    _nearest_neighbors[algorithm] = nearest_neighbors
    return nearest_neighbors


@register
def _knn(
    X: np.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import sklearn.neighbors
    except ImportError:
        raise "'scikit-learn' is not installed"
    index = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='auto',
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
    X: np.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import annoy
    except ImportError:
        raise "'Annoy' is not installed"
    n_samples, n_features = X.shape
    n_trees = kwargs.get('n_trees', 20)
    index = annoy.AnnoyIndex(n_features, metric=metric)
    if random_state is not None:
        index.set_seed(random_state)
    [index.add_item(i, x) for i, x in enumerate(X)]
    if n_jobs is not None:
        index.build(n_trees=n_trees, n_jobs=n_jobs)
    else:
        index.build(n_trees=n_trees, n_jobs=-1)
    neighbor_distances = np.empty((n_samples, n_neighbors), dtype=np.float64)
    neighbor_indices = np.empty((n_samples, n_neighbors), dtype=np.int64)
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
    X: np.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pynndescent
    except ImportError:
        raise "'PyNNDescent' is not installed, run `pip install manifold[pynndescent]` to install"
    n_samples = X.shape[0]
    n_trees = min(64, 5 + round(n_samples**0.5 / 20.0))
    n_iters = max(5, round(np.log2(n_samples)))
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
    X: np.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state=None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import scann
    except ImportError:
        raise "'ScaNN' is not installed, run `pip install manifold[scann]` to install"
    if not include_self:
        n_neighbors += 1
    index = scann.scann_ops_pybind.builder(
        X,
        n_neighbors,
        metric,
    ).tree(
        num_leaves=2000,
        num_leaves_to_search=100,
        training_sample_size=250000,
    ).score_ah(
        2,
        anisotropic_quantization_threshold=0.2,
    ).reorder(100).build()
    neighbor_indices, neighbor_distances = index.search_batched(
        n_neighbors,
        leaves_to_search=150,
    )
    if not include_self:
        neighbor_indices = neighbor_indices[:, 1:]
        neighbor_distances = neighbor_distances[:, 1:]
    return neighbor_indices, neighbor_distances


def nearest_neighbors(
    X,
    n_neighbors: int,
    *,
    algorithm: str = 'annoy',
    metric: str = 'euclidean',
    metric_params: Optional[dict] = None,
):
    if algorithm not in _nearest_neighbors:
        raise ValueError(f"'{algorithm}' is not supported")

    return _nearest_neighbors[algorithm]()


def neighbor_graph(
    nearest_neighbors,
    dense: bool = False,
) -> torch.Tensor:
    neighbor_indices, neighbor_distances = nearest_neighbors
    n_samples, n_neighbors = neighbor_indices
    crow_indices = torch.arange(0, n_samples * n_neighbors + 1, n_neighbors)
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
    if dense:
        return neighbor_connectivities.to_dense(), neighbor_distances.to_dense()
    return neighbor_connectivities, neighbor_distances


def neighbor_pair_indices(neighbor_graph):
    """
    Return:
        neighbor samples (B * n_samples, 2)
    """
    neighbor_connectivities, _ = neighbor_graph
    neighbor_connectivities = neighbor_connectivities.to_sparse_coo()
    return neighbor_connectivities.indices().T


def neighbor_samples(
    neighbor_pairs,
    n_negative_samples: int = 5,
):
    return
