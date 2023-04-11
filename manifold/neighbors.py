from typing import Tuple, Optional

import numpy
import torch

from .base import Estimator

_nearest_neighbors = {}


def register(nearest_neighbors):
    algorithm = nearest_neighbors.__name__[1:]
    if algorithm in _nearest_neighbors:
        raise ValueError(f"'{algorithm}' is already registered")
    _nearest_neighbors[algorithm] = nearest_neighbors
    return nearest_neighbors


@register
def _knn(
    X: numpy.ndarray,
    n_neighbors: int,
    *,
    include_self: bool = False,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    try:
        import sklearn.neighbors
    except ImportError:
        raise ImportError("'scikit-learn' is not installed")
    index = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='auto',
        metric=metric,
        metric_kwargs=metric_kwargs,
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
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    try:
        import annoy
    except ImportError:
        raise ImportError("'Annoy' is not installed")
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
    neighbor_distances = numpy.empty((n_samples, n_neighbors),
                                     dtype=numpy.float64)
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
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    try:
        import pynndescent
    except ImportError:
        raise ImportError(
            "'PyNNDescent' is not installed, run `pip install manifold[pynndescent]` to install"
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
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    random_state=None,
    **kwargs,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    try:
        import scann
    except ImportError:
        raise ImportError(
            "'ScaNN' is not installed, run `pip install manifold[scann]` to install"
        )
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
    X: torch.Tensor,
    n_neighbors: int,
    *,
    algorithm: str = 'annoy',
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = X.detach().cpu().numpy()
    if algorithm not in _nearest_neighbors:
        raise ValueError(f"'{algorithm}' is not supported")
    neighbor_indices, neighbor_distances = _nearest_neighbors[algorithm](
        X,
        n_neighbors,
        metric=metric,
        metric_kwargs=metric_kwargs,
    )
    return torch.from_numpy(neighbor_indices), torch.from_numpy(
        neighbor_distances)


def neighbor_graph(
    X: Optional[torch.Tensor] = None,
    n_neighbors: Optional[int] = None,
    *,
    algorithm: str = 'annoy',
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    nearest_neighbors_: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if nearest_neighbors_ is None:
        nearest_neighbors_ = nearest_neighbors(
            X,
            n_neighbors,
            algorithm=algorithm,
            metric=metric,
            metric_kwargs=metric_kwargs,
        )
    neighbor_indices, neighbor_distances = nearest_neighbors_
    n_samples, n_neighbors = neighbor_indices.shape
    crow_indices = torch.arange(
        0,
        n_samples * n_neighbors + 1,
        n_neighbors,
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


def neighbor_pairs(
    X: Optional[torch.Tensor] = None,
    n_neighbors: Optional[int] = None,
    *,
    algorithm: str = 'annoy',
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    neighbor_graph_: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    neighbor_pairs_: Optional[str] = None,
) -> torch.Tensor:
    if neighbor_pairs_ is not None:
        return torch.load(neighbor_pairs_)
    if neighbor_graph_ is None:
        neighbor_graph_ = neighbor_graph(
            X,
            n_neighbors,
            algorithm=algorithm,
            metric=metric,
            metric_kwargs=metric_kwargs,
        )
    neighbor_connectivities, _ = neighbor_graph_
    return neighbor_connectivities.to_sparse_coo().indices().T


def neighbor_pair_loader(
    neighbor_pairs: torch.Tensor,
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
):
    return NeighborPairLoader(
        neighbor_pairs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


class NeighborPairLoader:

    def __init__(
        self,
        neighbor_pairs: torch.Tensor,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.neighbor_pairs = neighbor_pairs
        self.n_samples = self.neighbor_pairs.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        n_batches, remainder = divmod(self.n_samples, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.n_samples)
        else:
            self.indices = None
        self.index = 0
        return self

    def __next__(self):
        if self.index >= min(self.n_batches * self.batch_size, self.n_samples):
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.index:self.index + self.batch_size]
            batch = torch.index_select(self.neighbor_pairs, 0, indices)
        else:
            batch = self.neighbor_pairs[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch


# (2 * B, d)
def batch_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.vstack((embeddings[:, 0, :], embeddings[:, 1, :]))


# (B, 1) torch.int64
def batch_positive_samples(
    batch_size: int,
    num_positive_samples: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if num_positive_samples == 1:
        return torch.arange(
            batch_size,
            2 * batch_size,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(dim=1)
    raise NotImplementedError


def batch_max_negative_samples(
    batch_size: int,
    include_anchor: bool = False,
    include_positive: bool = False,
) -> int:
    max1 = batch_size if include_anchor else batch_size - 1
    max2 = batch_size if include_positive else batch_size - 1
    return max1 + max2


# (B, 2 * B) torch.bool
def batch_negative_sample_mask(
    batch_size: int,
    include_anchor: bool = False,
    include_positive: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    excluded = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    included = torch.ones_like(excluded)
    mask1 = included if include_anchor else excluded
    mask2 = included if include_positive else excluded
    return torch.hstack((mask1, mask2))


def batch_negative_sampling(
    batch_size: int,
    num_negative_samples: int = 5,
    include_anchor: bool = False,
    include_positive: bool = False,
    negative_sampling: str = 'uniform',
    replacement: bool = False,
    device: Optional[torch.device] = None,
) -> dict:
    max_negative_samples = batch_max_negative_samples(
        batch_size,
        include_anchor=include_anchor,
        include_positive=include_positive,
    )
    num_negative_samples = min(num_negative_samples, max_negative_samples)
    if negative_sampling == 'uniform':
        negative_sample_mask = batch_negative_sample_mask(
            batch_size,
            include_anchor=include_anchor,
            include_positive=include_positive,
        )
        weights = negative_sample_mask.to(dtype=torch.float64)
    else:
        raise NotImplementedError
    return {
        'num_negative_samples': num_negative_samples,
        'weights': weights,
        'replacement': replacement,
        'device': device,
    }


# (B, m) torch.int64
def batch_negative_samples(
    batch_size: Optional[int] = None,
    num_negative_samples: int = 5,
    include_anchor: bool = False,
    include_positive: bool = False,
    negative_sampling: str = 'uniform',
    weights: Optional[torch.Tensor] = None,
    replacement: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if weights is None:
        kwargs = batch_negative_sampling(
            batch_size,
            num_negative_samples=num_negative_samples,
            include_anchor=include_anchor,
            include_positive=include_positive,
            negative_sampling=negative_sampling,
            replacement=replacement,
            device=device,
        )
        num_negative_samples = kwargs['num_negative_samples']
        weights = kwargs['weights']
    return torch.multinomial(
        weights,
        num_negative_samples,
        replacement=replacement,
    ).to(device=device)


# (B, 1) torch.int64
def batch_anchor_index(
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.arange(
        0,
        batch_size,
        dtype=torch.int64,
        device=device,
    ).unsqueeze(dim=1)


# (B, 1 + m) torch.int64
def batch_neighbor_indices(
    batch_size: Optional[int] = None,
    num_positive_samples: int = 1,
    num_negative_samples: int = 5,
    include_anchor: bool = False,
    include_positive: bool = False,
    negative_sampling: str = 'uniform',
    weights: Optional[torch.Tensor] = None,
    replacement: bool = False,
    device: Optional[torch.device] = None,
    positive_samples: Optional[torch.Tensor] = None,
    negative_samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if positive_samples is None:
        positive_samples = batch_positive_samples(
            batch_size,
            num_positive_samples=num_positive_samples,
            device=device,
        )
    if negative_samples is None:
        negative_samples = batch_negative_samples(
            batch_size=batch_size,
            num_negative_samples=num_negative_samples,
            include_anchor=include_anchor,
            include_positive=include_positive,
            negative_sampling=negative_sampling,
            weights=weights,
            replacement=replacement,
            device=device,
        )
    return torch.hstack((positive_samples, negative_samples))


# (B, 1 + m) torch.bool
def batch_neighbor_mask(
    batch_size: Optional[int] = None,
    num_positive_samples: int = 1,
    num_negative_samples: int = 5,
    device: Optional[torch.device] = None,
    neighbor_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if neighbor_indices is not None:
        neighbor_mask = torch.zeros_like(neighbor_indices, dtype=torch.bool)
        neighbor_mask[:, :num_positive_samples] = True
    else:
        neighbor_mask = torch.zeros(
            (batch_size, num_positive_samples + num_negative_samples),
            dtype=torch.bool,
            device=device,
        )
        neighbor_mask[:, :num_positive_samples] = True
    return neighbor_mask


# torch.nn.Module interface


class NearestNeighbors(Estimator):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> 'NearestNeighbors':
        """Fit the model with X."""
        return self

    def fit_transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **fit_params,
    ) -> torch.Tensor:
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y=y, **fit_params).transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply dimensionality reduction to X."""
        return
