from typing import Optional

import torch

import manifold.neighbors
import manifold.metrics
import manifold.encoders
import manifold.optim
from manifold.base import Estimator
from .pca import pca

# functional interface


def loss(metric: str = 'mle', reduction: str = 'sum', **kwargs):
    if metric == 'mle':
        return manifold.metrics.MLELoss(reduction=reduction, **kwargs)
    if metric == 'nce':
        return manifold.metrics.NCELoss(reduction=reduction, **kwargs)
    if metric == 'infonce':
        return manifold.metrics.InfoNCELoss(reduction=reduction, **kwargs)
    raise ValueError(f"'{metric}' metric is not supported")


def embedding(
    X: torch.Tensor,
    n_components: int = 2,
    parametric: bool = False,
    init: str = 'pca',
    encoder: str = 'mlp',
    encoder_kwargs: dict = {},
    pretrained_model: Optional[str] = None,
) -> torch.nn.Module:
    N, D = X.shape
    if parametric:
        encoder = manifold.encoders.encoder(
            encoder,
            D,
            n_components,
            **encoder_kwargs,
        ),
        model = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(X, freeze=True),
            encoder,
        )
        if pretrained_model is not None:
            model.load_state_dict(torch.load(pretrained_model))
    else:
        if init == 'pca':
            Y = pca(X, n_components=n_components)
        elif init == 'random':
            Y = torch.randn(N, n_components)
        elif init == 'pretrained':
            Y = torch.load(pretrained_model)
        else:
            raise ValueError(f"'{init}' initiation is not supported")
        model = torch.nn.Embedding.from_pretrained(Y, freeze=False)
    return model


def log_partition(
    Z: float,
    learnable: bool = False,
) -> torch.Tensor:
    log_Z = torch.tensor(Z).log()
    if learnable:
        log_Z = torch.nn.Parameter(log_Z, requires_grad=True)
    return log_Z


def parameters(
    model: torch.nn.Module,
    log_Z: torch.Tensor,
    use_learnable_partition: bool = False,
    verbose: bool = False,
):
    params = [{'params': model.parameters()}]
    if use_learnable_partition:
        params.append({'params': log_Z, 'lr': 1e-3})
    # TODO: print the number of learnable parameters
    if verbose:
        raise NotImplementedError
    return params


def high_dimensional_similarity(
    neighbor: Optional[torch.Tensor] = None,
    anchor: Optional[torch.Tensor] = None,
    metric: str = 'euclidean',
    metric_kwargs: dict = {},
    neighbor_mask: Optional[torch.Tensor] = None,
    binary: bool = True,
) -> torch.Tensor:
    if binary:
        return neighbor_mask
    raise NotImplementedError


def low_dimensional_similarity(
    neighbor: torch.Tensor,
    anchor: torch.Tensor,
    metric: str = 'euclidean',
    metric_kwargs: dict = {},
) -> torch.Tensor:
    if metric == 'euclidean':
        distance = manifold.metrics.squared_euclidean_distance(neighbor, anchor)
        return manifold.metrics.cauchy_kernel(distance, **metric_kwargs)
    if metric == 'cosine':
        similarity = manifold.metrics.cosine_similarity(neighbor, anchor)
        return manifold.metrics.exp(similarity, **metric_kwargs)
    raise ValueError(f"'{metric}' metric is not supported")


def gradient_descent(
    epoch: int,
    dataloader,
    model: torch.nn.Module,
    log_Z: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: bool = True,
    clip_value: float = 4.0,
    P: Optional[torch.Tensor] = None,
    anchor_index: Optional[torch.Tensor] = None,
    neighbor_indices: Optional[torch.Tensor] = None,
    force_resampling: bool = False,
    positive_samples: Optional[torch.Tensor] = None,
    negative_sampling_kwargs: dict = {},
    num_positive_samples: int = 1,
    num_negative_samples: int = 5,
    include_anchor: bool = False,
    include_positive: bool = False,
    negative_sampling: str = 'uniform',
    replacement: bool = False,
    verbose: bool = False,
):
    losses = []
    model.train()
    device = next(model.parameters()).device
    use_binary_high_dimensional_similarity = (P is not None)
    size = len(dataloader)
    batch_size = len(anchor_index)
    for index, neighbor_pairs in enumerate(dataloader):
        # handle the last incomplete batch
        B, _ = neighbor_pairs.shape
        if B != batch_size:
            if index != size - 1:
                raise RuntimeError('expected the last batch')
            anchor_index = manifold.neighbors.batch_anchor_index(
                B,
                device=device,
            )
            neighbor_indices = manifold.neighbors.batch_neighbor_indices(
                B,
                num_positive_samples=num_positive_samples,
                num_negative_samples=num_negative_samples,
                include_anchor=include_anchor,
                include_positive=include_positive,
                negative_sampling=negative_sampling,
                replacement=replacement,
                device=device,
            )
            force_resampling = False
            if use_binary_high_dimensional_similarity:
                neighbor_mask = manifold.neighbors.batch_neighbor_mask(
                    B,
                    num_positive_samples=num_positive_samples,
                    num_negative_samples=num_negative_samples,
                    device=device,
                )
                P = high_dimensional_similarity(neighbor_mask=neighbor_mask)
        # look up embeddings
        neighbor_pairs = neighbor_pairs.to(device=device)
        embeddings = model(neighbor_pairs)
        embeddings = manifold.neighbors.batch_embeddings(embeddings)
        # negative sampling using cached variables
        if neighbor_indices is None or force_resampling:
            neighbor_indices = manifold.neighbors.batch_neighbor_indices(
                positive_samples=positive_samples,
                **negative_sampling_kwargs,
            )
        # TODO: look up high dimensional similarity
        if not use_binary_high_dimensional_similarity:
            raise NotImplementedError
        # compute low dimensional similarity
        anchor_embedding = embeddings[anchor_index]
        neighbor_embeddings = embeddings[neighbor_indices]
        Q = low_dimensional_similarity(neighbor_embeddings, anchor_embedding)
        # compute loss
        loss = criterion(P, Q, log_Z)
        # update parameters
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            if log_Z is not None:
                torch.nn.utils.clip_grad_value_(log_Z, clip_value)
        optimizer.step()
        # TODO: track metrics
        losses.append(loss.item())
        # TODO:
        if verbose:
            raise NotImplementedError
    epoch_loss = sum(losses) / len(losses)
    print(f'[epoch {epoch}] loss: {epoch_loss}')
    return epoch_loss


def tsne(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    n_components: int = 2,
    *,
    parametric: bool = False,
    encoder: str = 'mlp',
    encoder_kwargs: dict = {},
    init: str = 'pca',
    n_epochs: int = 100,
    batch_size: int = 1024,
    shuffle: bool = True,
    drop_last: bool = False,
    Z: float = 1.0,
    use_learnable_partition: bool = False,
    criterion: str = 'nce',
    reduction: str = 'sum',
    criterion_kwargs: dict = {},
    lr: float = 200.0,
    optimizer: str = 'sgd',
    optimizer_kwargs: dict = {},
    annealing: str = 'cosine',
    lr_scheduler: str = 'warm_restarts',
    lr_scheduler_kwargs: dict = {
     # 'T_max': 10,
        'T_0': 50,
        'T_mult': 1,
        'eta_min': 0.1,
    },
    include_anchor: bool = False,
    include_positive: bool = False,
    negative_sampling: str = 'uniform',
    replacement: bool = False,
    clip_grad: bool = True,
    clip_value: float = 4.0,
    n_neighbors: int = 5,
    nearest_neighbors: str = 'annoy',
    metric: str = 'euclidean',
    metric_kwargs: dict = {},
    use_binary_high_dimensional_similarity: bool = True,
    num_positive_samples: int = 1,
    num_negative_samples: int = 5,
    force_resampling: bool = False,
    autosave: bool = False,
    neighbor_graph: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    neighbor_pairs = manifold.neighbors.neighbor_pairs(
        X,
        n_neighbors,
        algorithm=nearest_neighbors,
        metric=metric,
        metric_kwargs=metric_kwargs,
        neighbor_pairs_=neighbor_graph,
    )
    # dataloader
    dataloader = manifold.neighbors.neighbor_pair_loader(
        neighbor_pairs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    # model
    model = embedding(
        X,
        n_components,
        parametric=parametric,
        init=init,
        encoder=encoder,
        encoder_kwargs=encoder_kwargs,
        pretrained_model=pretrained_model,
    )
    model.to(device=device)
    log_Z = log_partition(Z, learnable=use_learnable_partition)
    log_Z.to(device=device)
    # parameters
    params = parameters(
        model,
        log_Z,
        use_learnable_partition=use_learnable_partition,
        verbose=verbose,
    )
    # criterion
    criterion = loss(
        criterion,
        reduction=reduction,
        **criterion_kwargs,
    )
    # optimizer
    optimizer = manifold.optim.optimizer(
        optimizer,
        params,
        lr=lr,
        **optimizer_kwargs,
    )
    # learning rate scheduler
    lr_scheduler = manifold.optim.lr_scheduler(
        lr_scheduler,
        optimizer,
        annealing=annealing,
        **lr_scheduler_kwargs,
    )
    # cache batch size dependent variables
    anchor_index = manifold.neighbors.batch_anchor_index(
        batch_size,
        device=device,
    )
    positive_samples = manifold.neighbors.batch_positive_samples(
        batch_size,
        num_positive_samples=num_positive_samples,
        device=device,
    )
    negative_sampling_kwargs = manifold.neighbors.batch_negative_sampling(
        batch_size,
        num_negative_samples=num_negative_samples,
        include_anchor=include_anchor,
        include_positive=include_positive,
        negative_sampling=negative_sampling,
        replacement=replacement,
        device=device,
    )
    if use_binary_high_dimensional_similarity:
        neighbor_mask = manifold.neighbors.batch_neighbor_mask(
            batch_size,
            num_positive_samples=num_positive_samples,
            num_negative_samples=num_negative_samples,
            device=device,
        )
        P = high_dimensional_similarity(neighbor_mask=neighbor_mask)
    else:
        P = None
    # training
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_loss = gradient_descent(
            epoch,
            dataloader=dataloader,
            model=model,
            log_Z=log_Z,
            P=P,
            criterion=criterion,
            optimizer=optimizer,
            clip_grad=clip_grad,
            clip_value=clip_value,
            anchor_index=anchor_index,
            force_resampling=force_resampling,
            positive_samples=positive_samples,
            negative_sampling_kwargs=negative_sampling_kwargs,
            num_positive_samples=num_positive_samples,
            num_negative_samples=num_negative_samples,
            include_anchor=include_anchor,
            include_positive=include_positive,
            negative_sampling=negative_sampling,
            replacement=replacement,
            verbose=verbose,
        )
        lr_scheduler.step()
        epoch_losses.append(epoch_loss)
    if autosave:
        torch.save(model.state_dict(), model)
    return model.weight.detach().cpu(), epoch_losses


# torch.nn.Module interface


class TSNE(Estimator):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> 'TSNE':
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
