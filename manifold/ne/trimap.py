import os
from typing import Optional, Tuple

import torch

import manifold.encoders
import manifold.metrics
import manifold.neighbors
import manifold.optim
import manifold.samplers
from manifold.base import Estimator

from .pca import pca

# functional interface


def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def preprocessing(
    X: torch.Tensor,
    apply_pca: bool = True,
    max_features: int = 100,
    normalizing_dim: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Preprocess X."""
    _, num_features = X.shape
    applied_pca = False
    if num_features > max_features and apply_pca:
        vprint(
            verbose, f"truncating to {max_features} features by applying PCA"
        )
        # expected applying mean normalization first when applying PCA
        # expected the same dtype, device as X
        X = pca(X, max_features)
        applied_pca = True
    else:
        vprint(verbose, f"scaling {num_features} features to the range [0, 1]")
        # expected the same dtype, device as X
        X = manifold.metrics.minmax_normalize(X, dim=normalizing_dim)
        # expected the same dtype, device as X
        X = manifold.metrics.mean_normalize(X, dim=0)
    return X, applied_pca


def neighboring(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    num_neighbors: int = 10,
    *,
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
    autosave: bool = True,
    root: str = "./",
    nearest_neighbors: Optional[str] = None,
    neighboring_samples: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[manifold.samplers.DataLoader, manifold.samplers.Sampler]:
    """Based on the nearest neighbors for each sample in X, returns a dataloader
    that iterates over the selected neighboring samples, together with a sampler
    that draws anchor and neighbor samples for each batch.
    """
    # expected a file for neighboring_samples
    if neighboring_samples is not None:
        neighboring_samples = os.path.join(root, neighboring_samples)
    if neighboring_samples is None or (not os.path.exists(neighboring_samples)):
        # expected a file for nearest neighbors
        if nearest_neighbors is not None:
            nearest_neighbors = os.path.join(root, nearest_neighbors)
        if nearest_neighbors is None or (not os.path.exists(nearest_neighbors)):
            vprint(verbose, "searching nearest neighbors")
            neighbors = manifold.neighbors.nearest_neighbors(
                X,
                num_neighbors,
                algorithm=algorithm,
                metric=metric,
                metric_kwargs=metric_kwargs,
                generator=generator,
                device=device,
            )
            if nearest_neighbors is not None and autosave:
                vprint(
                    verbose,
                    f"autosaving nearest neighbors to {nearest_neighbors}",
                )
                neighbor_indices, neighbor_distances = neighbors
                torch.save(
                    (neighbor_indices.cpu(), neighbor_distances.cpu()),
                    nearest_neighbors,
                )
        else:
            vprint(
                verbose,
                f"loading nearest neighbors from {nearest_neighbors}",
            )
            neighbor_indices, neighbor_distances = torch.load(nearest_neighbors)
            neighbors = (
                neighbor_indices.to(device=device),
                neighbor_distances.to(device=device),
            )
        if scale_nearest_neighbors:
            vprint(
                verbose,
                "scaling nearest neighbors",
            )
            neighbors = manifold.neighbors.scale(
                neighbors,
                start=scaling_start,
                end=scaling_end,
            )
        neighbor_indices, _ = neighbors
        vprint(
            verbose,
            "drawing neighboring samples, i.e., pairs, triplets, or larger neighborhoods",
        )
        sampler = manifold.samplers.neighbor_sampler(
            num_positive_samples,
            num_negative_samples,
            force_resampling=force_resampling,
            exclude_anchor_samples=exclude_anchor_samples,
            exclude_positive_samples=exclude_positive_samples,
            exclude_neighbors=exclude_neighbors,
            replacement=replacement,
            generator=generator,
            device=device,
        )
        samples = sampler(neighbor_indices)
        if neighboring_samples is not None and autosave:
            vprint(
                verbose,
                f"autosaving neighboring samples to {neighboring_samples}",
            )
            torch.save(samples.cpu(), neighboring_samples)
    else:
        vprint(
            verbose,
            f"loading neighboring samples from {neighboring_samples}",
        )
        samples = torch.load(neighboring_samples).to(device=device)
    vprint(verbose, "constructing dataloader for neighboring samples")
    dataloader = manifold.samplers.NeighborLoader(
        samples,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        generator=generator,
        device=device,
    )
    vprint(verbose, "constructing sampler for negative sampling in each batch")
    # fixme: access number of positives (and negatives) rather than desired one, they might be different
    # num_positive_samples = sampler.num_positive_samples
    # if not force_resampling:
    #     num_negative_samples = sampler.num_positive_samples
    batch_sampler = manifold.samplers.batch_neighbor_sampler(
        num_positive_samples,
        num_negative_samples,
        force_resampling=force_resampling,
        in_batch=in_batch,
        exclude_anchor_samples=exclude_anchor_samples,
        exclude_positive_samples=exclude_positive_samples,
        exclude_neighbors=exclude_neighbors,
        replacement=replacement,
        generator=generator,
        device=device,
    )
    return dataloader, batch_sampler


def embedding(
    X: torch.Tensor,
    num_components: int = 2,
    *,
    parametric: bool = False,
    init: str = "pca",
    applied_pca: bool = False,
    encoder: str = "mlp",
    encoder_kwargs: dict = {},
    root: str = "./",
    pretrained_model: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """Returns a lookup table that stores the embeddings of X in low dimensional
    space.
    """
    num_samples, num_features = X.shape
    if parametric:
        vprint(verbose, "parametric embedding")
        encoder = (
            manifold.encoders.encoder(
                encoder,
                num_features,
                num_components,
                **encoder_kwargs,
            ),
        )
        model = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(X, freeze=True),
            encoder,
        )
        if pretrained_model is not None:
            pretrained_model = os.path.join(root, pretrained_model)
            vprint(verbose, f"loading pretrained model from {pretrained_model}")
            model.load_state_dict(torch.load(pretrained_model))
    else:
        # expected the same dtype as X
        if init == "pca":
            vprint(
                verbose,
                "initializing embeddings in low dimensional space by applying PCA",
            )
            Y = pca(X, num_components)
            if applied_pca:
                # expected num_components < max_features
                Y = X[:, :num_components]
        elif init == "random":
            vprint(
                verbose,
                "randomly initializing embeddings in low dimensional space",
            )
            Y = torch.randn(
                num_samples,
                num_components,
                dtype=X.dtype,
                generator=generator,
            )
        elif init == "pretrained":
            pretrained_model = os.path.join(root, pretrained_model)
            vprint(
                verbose,
                f"loading pretrained embeddings from {pretrained_model}",
            )
            Y = torch.load(pretrained_model)
        else:
            raise ValueError(f"initialization '{init}' is not supported")
        model = torch.nn.Embedding.from_pretrained(Y, freeze=False)
    if device is not None:
        model = model.to(device=device)
    return model


def log_partition(
    partition: float,
    learnable: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns the log partition function."""
    log_Z = torch.tensor(partition, dtype=dtype, device=device).log()
    if learnable:
        log_Z = torch.nn.Parameter(log_Z, requires_grad=True)
    return log_Z


def parameters(
    model: torch.nn.Module,
    use_learnable_partition: bool = False,
    log_partition: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> list:
    """Returns a list of parameters."""
    params = [{"params": model.parameters()}]
    if use_learnable_partition:
        # expected constant learning rate for log partition function
        params.append({"params": log_partition, "lr": 1e-3})
    if verbose:
        num_params = sum(params.numel() for params in model.parameters())
        if use_learnable_partition:
            num_params += 1
        vprint(verbose, f"total number of learnable parameters: {num_params}")
    return params


def loss(
    model: torch.nn.Module,
    neighboring_samples: torch.Tensor,
    sampler: manifold.samplers.Sampler,
    force_resampling: bool = False,
    metric: str = "euclidean",
    verbose: bool = False,
) -> torch.Tensor:
    """Returns the loss for TriMAP."""
    anchor_indices, neighbor_indices = sampler(
        neighboring_samples,
        force_resampling=force_resampling,
    )
    sampler.embedding = model
    anchors = sampler.embeddings(anchor_indices)
    neighbors = sampler.embeddings(neighbor_indices)
    neighbor_sampling_mask = sampler.neighbor_sampling_mask
    P = manifold.metrics.similarity(
        metric="binary", mask=neighbor_sampling_mask
    )
    Q = manifold.metrics.similarity(neighbors, anchors, metric=metric)
    # vprint(
    #     verbose,
    #     f"P: {P.shape}, Q: {Q.shape}, mask: {neighbor_sampling_mask.shape}",
    # )
    return manifold.metrics.infonce_loss(
        P,
        Q,
        reduction="sum",
        with_exp=False,
        with_log=False,
    )


def train(
    epoch: int,
    dataloader: manifold.samplers.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: manifold.samplers.Sampler,
    force_resampling: bool = False,
    metric: str = "euclidean",
    clip_grad: bool = True,
    clip_value: float = 4.0,
    use_learnable_partition: bool = False,
    log_partition: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Trains the model."""
    losses = []
    model.train()
    for i, neighboring_samples in enumerate(dataloader):
        # vprint(verbose, "computing loss")
        batch_loss = loss(
            model,
            neighboring_samples,
            sampler,
            force_resampling=force_resampling,
            metric=metric,
            verbose=verbose,
        )
        # vprint(verbose, "updating parameters")
        optimizer.zero_grad()
        batch_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            if use_learnable_partition:
                torch.nn.utils.clip_grad_value_(log_partition, clip_value)
        optimizer.step()
        # fixme: track metrics
        losses.append(batch_loss.item())
    epoch_loss = sum(losses) / len(losses)
    vprint(verbose, f"[epoch {epoch}] loss: {epoch_loss}")
    return epoch_loss


def autosaving(
    model: torch.nn.Module,
    parametric: bool = False,
    autosave: bool = True,
    root: str = "./",
    trained_model: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Autosaves the model."""
    if trained_model is not None and autosave:
        trained_model = os.path.join(root, trained_model)
        if parametric:
            vprint(verbose, f"autosaving trained model to {trained_model}")
            torch.save(model.state_dict(), trained_model)
        else:
            vprint(verbose, f"autosaving trained embeddings to {trained_model}")
            torch.save(model.weight.detach().cpu(), trained_model)


def trimap(
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
    lr: float = 200.0,
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
    # device
    if device is None:
        # torch.set_default_tensor_type()
        # torch.set_default_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vprint(verbose, f"running on '{device}' device")
    # X
    applied_pca = False
    if preprocess:
        X, applied_pca = preprocessing(
            X,
            apply_pca=apply_pca,
            max_features=max_features,
            normalizing_dim=normalizing_dim,
            verbose=verbose,
        )
    # dataloader, sampler
    dataloader, sampler = neighboring(
        X,
        y,
        num_neighbors,
        algorithm=algorithm,
        metric=metric,
        metric_kwargs=metric_kwargs,
        scale_nearest_neighbors=scale_nearest_neighbors,
        scaling_start=scaling_start,
        scaling_end=scaling_end,
        num_positive_samples=num_positive_samples,
        num_negative_samples=num_negative_samples,
        force_resampling=force_resampling,
        in_batch=in_batch,
        exclude_anchor_samples=exclude_anchor_samples,
        exclude_positive_samples=exclude_positive_samples,
        exclude_neighbors=exclude_neighbors,
        replacement=replacement,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        autosave=autosave,
        root=root,
        nearest_neighbors=nearest_neighbors,
        neighboring_samples=neighboring_samples,
        generator=generator,
        device=device,
        verbose=verbose,
    )
    # model
    model = embedding(
        X,
        num_components,
        parametric=parametric,
        init=init,
        applied_pca=applied_pca,
        encoder=encoder,
        encoder_kwargs=encoder_kwargs,
        root=root,
        pretrained_model=pretrained_model,
        generator=generator,
        device=device,
        verbose=verbose,
    )
    # log partition
    log_Z = log_partition(
        partition,
        learnable=use_learnable_partition,
        dtype=X.dtype,
        device=device,
    )
    # parameters
    params = parameters(
        model,
        use_learnable_partition=use_learnable_partition,
        log_partition=log_Z,
        verbose=verbose,
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
    # training
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = train(
            epoch,
            dataloader,
            model,
            optimizer,
            sampler,
            force_resampling=force_resampling,
            metric=metric,
            clip_grad=clip_grad,
            clip_value=clip_value,
            use_learnable_partition=use_learnable_partition,
            log_partition=log_Z,
            verbose=verbose,
        )
        lr_scheduler.step()
        epoch_losses.append(epoch_loss)
    # Y
    vprint(verbose, "computing embeddings in low dimensional space")
    if parametric:
        model.eval()
        num_samples, _ = X.shape
        anchors = torch.arange(num_samples, dtype=torch.int64, device=device)
        Y = model(anchors).detach().cpu()
    else:
        Y = model.weight.detach().cpu()
    # autosaving
    autosaving(
        model,
        parametric=parametric,
        autosave=autosave,
        root=root,
        trained_model=trained_model,
        verbose=verbose,
    )
    return Y, epoch_losses


# class interface


class TriMAP(Estimator):
    """TriMAP.

    References:
        [1] https://github.com/eamid/trimap/tree/master
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
    ) -> "TriMAP":
        raise NotImplementedError
        # return self

    def fit_transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return trimap(
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
