from typing import Callable, Optional

import torch

import neighbors
import metrics
import datasets
import encoders
import optim
from .base import Estimator
from .pca import PCA


def _similarity_graph(
    neighbor_graph: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    *,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    eps: float = 1e-21,
    binary: bool = False,
    autosave: bool = True,
    similarity_graph: Optional[str] = None,
) -> torch.Tensor:
    if neighbor_graph is None and similarity_graph is not None:
        return torch.load(similarity_graph)
    if binary:
        P = neighbor_graph
    else:
        pass
    if autosave:
        torch.save(P, similarity_graph)
    return P


def _embedding(
    X: torch.Tensor,
    n_components: int = 2,
    init: str = 'pca',
    model: Optional[str] = None,
    parametric: bool = False,
    encoder: str = 'mlp',
    encoder_kwargs: dict = {},
) -> torch.nn.Module:
    """Returns initial neighbor embedding.

    Args:
        X (torch.Tensor)

    Shape:
        - X: :math:`(m, n)`

    """
    m, n = X.shape[-2:]
    if parametric:
        _model = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(X, freeze=True),
            encoders.encoder(
                n,
                n_components,
                _encoder=encoder,
                **encoder_kwargs,
            ),
        )
        if model is not None:
            _model.load_state_dict(torch.load(model))
    else:
        if init == 'pca':
            pca = PCA(n_components=n_components)
            Y = pca.fit_transform(X)
        elif init == 'random':
            Y = torch.randn(m, n_components)
        else:
            raise ValueError
        _model = torch.nn.Embedding.from_pretrained(Y, freeze=False)
    return _model


def _similarity_graph_embedded(
    neighbor_indices,
    embedding,
    metric: str = 'euclidean',
    metric_kwargs: Optional[dict] = None,
    eps: float = 1e-21,
):
    """
    Args:

    Shape:

    """
    samples = embedding(neighbor_indices)
    d = metrics.distance(
        samples[:, 0, :],
        samples[:, 1:, :],
        metric,
    )
    Q = metrics.cauchy_kernel(d)
    return Q


def _gradient_descent(
    epoch,
    neighbor_loader: torch.utils.data.Dataloader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: bool = True,
    clip_value: float = 4.0,
):
    model.train()
    for idx, neighbor_pair_indices in enumerate(neighbor_loader):
        neighbors = neighbors.to(next(model.parameters()).device)
        if force_resample:
            neighbor_indices = neighbors.neighbor_samples(
                neighbors,
                n_negative_samples,
            )
        P = _similarity_graph(neighbor_indices)
        Q = _similarity_graph_embedded(neighbor_indices)
        loss = criterion(P, Q)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()


def tsne(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    n_components: int = 2,
    *,
    parametric: bool = False,
    encoder: str = 'mlp',
    encoder_kwargs: dict = {},
    init: str = 'pca',
    n_epochs: int = 1000,
    batch_size: int = 1024,
    shuffle: bool = False,
    lr: float = 200.0,
    optimizer: str = 'sgd',
    optimizer_kwargs: dict = {},
    lr_scheduler: str = 'annealing',
    lr_scheduler_kwargs: dict = {},
    clip_grad: bool = True,
    clip_value: float = 4.0,
    n_neighbors: int = 5,
    nearest_neighbors: str = 'annoy',
    metric: str = 'euclidean',
    metric_kwargs: dict = {},
    binary_similarity_graph: bool = False,
    n_negative_samples: int = 5,
    force_resample: bool = False,
    autosave: bool = True,
    neighbor_graph: Optional[str] = None,
    similarity_graph: Optional[str] = None,
    model: Optional[str] = None,
) -> torch.Tensor:
    _neighbor_graph = neighbors.neighbor_graph(
        X,
        n_neighbors,
        algorithm=nearest_neighbors,
        metric=metric,
        metric_kwargs=metric_kwargs,
        autosave=autosave,
        neighbor_graph=neighbor_graph,
    )
    _neighbor_loader = datasets.neighbor_loader(
        _neighbor_graph,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    _model = _embedding(
        X,
        n_components,
        init=init,
        model=model,
        parametric=parametric,
        encoder=encoder,
        **encoder_kwargs,
    )
    _criterion = metrics.binary_cross_entropy
    _optimizer = optim.optimizer(
        _model.parameters(),
        lr,
        _optimizer=optimizer,
        **optimizer_kwargs,
    )
    _lr_scheduler = optim.lr_scheduler(
        optimizer,
        _lr_scheduler=lr_scheduler,
        **lr_scheduler_kwargs,
    )
    for epoch in range(n_epochs):
        _gradient_descent(
            epoch,
            _neighbor_loader,
            _model,
            criterion=_criterion,
            optimizer=_optimizer,
            clip_grad=clip_grad,
            clip_value=clip_value,
        )
        _lr_scheduler.step()
    if autosave:
        torch.save(_model.state_dict(), model)
    return _model.weight
