import os
import time
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

import manifold.encoders
import manifold.loggers
import manifold.metrics
import manifold.neighbors
import manifold.optim
import manifold.samplers
import manifold.transforms
from manifold.base import Estimator
from manifold.decorators import vtimeit

from .pca import pca

__all__ = (
    "vprint",
    "to_dtype",
    "to_device",
    "to_generator",
    "preprocess",
    "neighboring",
    "samples",
    "save",
    "load",
    "embedding",
    "log_partition",
    "parameters",
    "train",
    "evaluate",
    "fit",
    "transform",
    "loss",
    "tscne",
    "TSCNE",
)


# functional interface


def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def timestamp(format: str = "%b%d_%H-%M-%S") -> str:
    return time.strftime(format, time.localtime(time.time()))


def to_dtype(dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    """Returns the desired floating point.

    References:
        [1] https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
    """
    # expected torch.float
    if dtype is None:
        # torch.set_default_dtype(dtype)
        return torch.get_default_dtype()
    # expected half, float or double
    if isinstance(dtype, str):
        if dtype in ("half", "float16"):
            return torch.float16
        if dtype in ("float", "float32"):
            return torch.float32
        if dtype in ("double", "float64"):
            return torch.float64
    # expected torch.half, torch.float or torch.double
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"'{dtype}' is not supported to set floating point")


def to_device(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Returns the desired device.

    References:
        [1] https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    """
    if device is None:
        # torch.set_default_device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # expected cpu, cuda, cuda:0, cuda:1, etc.
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    raise ValueError(f"'{device}' is not supported to set 'torch.device'")


def to_generator(
    seed: Optional[Union[int, torch.Generator]] = None,
    device: Optional[torch.device] = None,
) -> torch.Generator:
    """Returns a generator for random sampling.

    References:
        [1] https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    if seed is None:
        return torch.Generator()
    if isinstance(seed, int):
        generator = torch.Generator(device)
        generator.manual_seed(seed)
        return generator
    if isinstance(seed, torch.Generator):
        return seed
    raise ValueError(f"'{seed}' is not supported to seed 'torch.Generator'")


@vtimeit
def preprocess(
    X: torch.Tensor,
    *,
    transform: Union[str, Sequence[str], Callable] = "pca",
    max_features: int = 100,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Preprocesses X and indicates whether PCA is applied.

    References:
        [1] https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    """
    # expected X as a torch.half, torch.float or torch.double tensor
    applied_pca = False
    if transform is None or transform == "none":
        return X, applied_pca
    if callable(transform):
        # expected the same dtype, device as original
        return transform(X), applied_pca
    _, num_features = X.shape
    if "image" in transform:
        vprint(verbose, f"scaling each pixel to [0, 1]")
        X = manifold.transforms.minmax_normalize(X)
    if "minmax_normalize" in transform:
        vprint(
            verbose,
            f"scaling {num_features} features to [0, 1]",
        )
        X = manifold.transforms.minmax_normalize(X, dim=0)
    if "mean_normalize" in transform:
        vprint(
            verbose,
            f"scaling {num_features} features to [-1, 1] with zero mean",
        )
        X = manifold.transforms.mean_normalize(X, dim=0)
    if "standardize" in transform:
        vprint(
            verbose,
            f"standardizing {num_features} features with zero mean and unit variance",
        )
        X = manifold.transforms.standardize(X, dim=0)
    if "pca" in transform:
        if num_features > max_features:
            vprint(
                verbose,
                f"truncating to {max_features} features by applying PCA",
            )
            X = pca(X, num_components=max_features)
            applied_pca = True
        else:
            vprint(verbose, f"centering {num_features} features with zero mean")
            X = manifold.transforms.center(X, dim=0)
    if "center" in transform:
        vprint(verbose, f"centering {num_features} features with zero mean")
        X = manifold.transforms.center(X, dim=0)
    # euclidean distance is equivalent to angular distance between any two samples
    if "l2_normalize" in transform:
        vprint(verbose, "applying L2 normalization to each sample")
        X = manifold.transforms.normalize(X, dim=1)
    return X, applied_pca


@vtimeit
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
    num_extra_neighbors: int = 50,
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
    dataset: str = "mnist",
    root: str = "./runs/",
    nearest_neighbors: Optional[str] = None,
    neighboring_samples: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[manifold.samplers.DataLoader, manifold.samplers.Sampler]:
    """Given the nearest neighbors for each sample in `X`, returns a `manifold.samplers.DataLoader`
    that iterates over the selected neighboring samples, together with a `manifold.samplers.Sampler`
    that draws anchor and neighbor samples for each batch.
    """
    # expect a file (path) for neighboring_samples
    if neighboring_samples is not None:
        neighboring_samples = os.path.join(root, neighboring_samples)
    if neighboring_samples is None or (not os.path.exists(neighboring_samples)):
        # expect a file (path) for nearest_neighbors
        if nearest_neighbors is not None:
            nearest_neighbors = os.path.join(root, nearest_neighbors)
        # expect num_extra_neighbors > 0
        if scale_nearest_neighbors and num_extra_neighbors is not None:
            vprint(verbose, f"using {num_extra_neighbors} extra neighbors")
            _num_neighbors = num_neighbors
            num_neighbors += num_extra_neighbors
        if nearest_neighbors is None or (not os.path.exists(nearest_neighbors)):
            vprint(verbose, "searching nearest neighbors")
            if verbose:
                since = time.time()
            neighbors = manifold.neighbors.nearest_neighbors(
                X,
                num_neighbors,
                algorithm=algorithm,
                metric=metric,
                metric_kwargs=metric_kwargs,
                generator=generator,
                device=device,
            )
            if verbose:
                print(
                    f"'manifold.neighbors.nearest_neighbors' executed, wall time: {time.time() - since:.4f} s"
                )
            if autosave:
                if nearest_neighbors is None:
                    nearest_neighbors = os.path.join(
                        root,
                        f"{dataset}_{num_neighbors}-nn_{metric}_{algorithm}.pt",
                    )
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
            vprint(verbose, "scaling nearest neighbors")
            if verbose:
                since = time.time()
            neighbors = manifold.neighbors.scale(
                neighbors,
                start=scaling_start,
                end=scaling_end,
            )
            if verbose:
                print(
                    f"'manifold.neighbors.scale' executed, wall time: {time.time() - since:.4f} s"
                )
            if num_extra_neighbors is not None:
                vprint(
                    verbose, f"truncating to {_num_neighbors} nearest neighbors"
                )
                neighbor_indices, neighbor_distances = neighbors
                neighbors = (
                    neighbor_indices[:, :_num_neighbors],
                    neighbor_distances[:, :_num_neighbors],
                )
        vprint(
            verbose,
            "drawing neighboring samples, i.e., pairs, triplets, or larger neighborhoods",
        )
        if verbose:
            since = time.time()
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
        neighbor_indices, _ = neighbors
        samples = sampler(neighbor_indices, labels=y)
        if verbose:
            print(
                f"neighboring samples drawed, wall time: {time.time() - since:.4f} s"
            )
        if autosave:
            if neighboring_samples is None:
                neighboring_samples = os.path.join(
                    root,
                    f"{dataset}_samples_{num_positive_samples}-p_{num_negative_samples}-n.pt",
                )
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
    # neighboar loader
    vprint(
        verbose,
        "constructing 'manifold.samplers.DataLoader' for neighboring samples",
    )
    if verbose:
        since = time.time()
    dataloader = manifold.samplers.neighbor_loader(
        samples,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        generator=generator,
        device=device,
    )
    if verbose:
        print(
            f"'manifold.samplers.neighbor_loader' executed, wall time: {time.time() - since:.4f} s"
        )
    vprint(verbose, "constructing sampler for negative sampling in each batch")
    # batch neighbor sampler
    vprint(
        verbose,
        "constructing 'manifold.samplers.Sampler' to draw anchor and neighbor samples for each batch",
    )
    # fixme: access number of positives (and negatives) rather than desired one, they might be different
    # num_positive_samples = sampler.num_positive_samples
    # perform negative sampling by slicing each batch
    # if not force_resampling:
    #     num_negative_samples = sampler.num_negative_samples
    if verbose:
        since = time.time()
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
    if verbose:
        print(
            f"'manifold.samplers.batch_neighbor_sampler' executed, wall time: {time.time() - since:.4f} s"
        )
    return dataloader, batch_sampler


def samples(
    X: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns the indices of samples in X."""
    num_samples, _ = X.shape
    return torch.arange(num_samples, dtype=torch.int64, device=device)


@vtimeit
def save(
    model: torch.nn.Module,
    file: str,
    *,
    root: str = "./runs/",
    verbose: bool = False,
) -> None:
    """Saves the trained model on CPU."""
    file = os.path.join(root, file)
    # parametric model
    if model.parametric:
        vprint(verbose, f"saving trained model to {file}")
        torch.save(model.state_dict(), file)
    # non-parametric model
    else:
        vprint(verbose, f"saving trained embeddings to {file}")
        torch.save(model.weight.detach().cpu(), file)


@vtimeit
def load(
    file: str,
    *,
    parametric: bool = False,
    model: Optional[torch.nn.Module] = None,
    root: str = "./runs/",
    verbose: bool = False,
) -> torch.nn.Module:
    """Loads the pretrained model on CPU."""
    file = os.path.join(root, file)
    # parametric model
    if parametric:
        vprint(verbose, f"loading pretrained model from {file}")
        model.load_state_dict(torch.load(file))
    # non-parametric model
    else:
        vprint(verbose, f"loading pretrained embeddings from {file}")
        model = torch.nn.Embedding.from_pretrained(
            torch.load(file),
            freeze=False,
        )
    return model


@vtimeit
def embedding(
    X: torch.Tensor,
    num_components: int = 2,
    *,
    parametric: bool = False,
    init: str = "pca",
    applied_pca: bool = False,
    encoder: str = "mlp",
    encoder_kwargs: dict = {},
    root: str = "./runs/",
    pretrained_model: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    snapshot: bool = False,
    samples: Optional[torch.Tensor] = None,
    logger: Optional[manifold.loggers.Logger] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """Returns a lookup table that stores the embeddings of X in low dimensional
    space.
    """
    # parametric model
    if parametric:
        vprint(verbose, "parametric embedding")
        _, num_features = X.shape
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
            model = load(
                pretrained_model,
                model=model,
                parametric=parametric,
                root=root,
                verbose=verbose,
            )
    # non-parametric model
    else:
        if init == "pca":
            vprint(verbose, "initializing embeddings by applying PCA")
            Y = pca(X, num_components)
            if applied_pca:
                # expected num_components < max_features
                Y = X[:, :num_components]
            model = torch.nn.Embedding.from_pretrained(Y, freeze=False)
        elif init == "random":
            vprint(verbose, "randomly initializing embeddings")
            num_samples, _ = X.shape
            Y = torch.randn(
                num_samples,
                num_components,
                dtype=X.dtype,
                generator=generator,
            )
            model = torch.nn.Embedding.from_pretrained(Y, freeze=False)
        elif init == "pretrained":
            model = load(
                pretrained_model,
                parametric=parametric,
                root=root,
                verbose=verbose,
            )
        else:
            raise ValueError(f"initialization '{init}' is not supported")
    # add an attribute `parametric` to the model
    setattr(model, "parametric", parametric)
    # force the desired floating point
    if dtype is not None:
        model = model.to(dtype=dtype)
    # force the desired device
    if device is not None:
        model = model.to(device=device)
    # snapshot the initial state
    if snapshot and logger is not None:
        evaluate(model, samples, epoch=-1, logger=logger)
    return model


def log_partition(
    partition: float,
    learnable: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns the log partition."""
    log_Z = torch.tensor(partition, dtype=dtype, device=device).log()
    if learnable:
        log_Z = torch.nn.Parameter(log_Z, requires_grad=True)
    return log_Z


def parameters(
    model: torch.nn.Module,
    *,
    use_learnable_partition: bool = False,
    log_partition: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> list:
    """Returns a list of parameters."""
    params = [{"params": model.parameters()}]
    # fixme: handle learnable log partition
    if use_learnable_partition:
        # expected constant learning rate for log partition function
        params.append({"params": log_partition, "lr": 1e-3})
    if verbose:
        num_params = sum(params.numel() for params in model.parameters())
        # fixme: handle learnable log partition
        if use_learnable_partition:
            num_params += 1
        print(f"total number of learnable parameters: {num_params}")
    return params


def loss(
    model: torch.nn.Module,
    criterion: manifold.metrics.Metric,
    sampler: manifold.samplers.Sampler,
    neighboring_samples: torch.Tensor,
    *,
    force_resampling: bool = False,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Returns t-CNE loss."""
    anchors, neighbors = sampler(
        neighboring_samples,
        force_resampling=force_resampling,
    )
    sampler.embedding = model
    anchors = sampler.embeddings(anchors)
    neighbors = sampler.embeddings(neighbors)
    mask = sampler.neighbor_sampling_mask
    P = manifold.metrics.similarity(metric="binary", mask=mask)
    Q = manifold.metrics.similarity(neighbors, anchors, metric=metric)
    return criterion(P, Q)


def train(
    model: torch.nn.Module,
    criterion: manifold.metrics.Metric,
    optimizer: manifold.optim.Optimizer,
    dataloader: manifold.samplers.DataLoader,
    sampler: manifold.samplers.Sampler,
    *,
    force_resampling: bool = False,
    metric: str = "euclidean",
    clip_grad: bool = True,
    clip_value: float = 4.0,
    use_learnable_partition: bool = False,
    log_partition: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
    logger: Optional[manifold.loggers.Logger] = None,
) -> None:
    """Trains the model one epoch."""
    model.train()
    num_steps = len(dataloader)
    for step, neighboring_samples in enumerate(dataloader):
        step_loss = loss(
            model,
            criterion,
            sampler,
            neighboring_samples,
            force_resampling=force_resampling,
            metric=metric,
        )
        optimizer.zero_grad()
        step_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            # fixme: handle learnable log partition
            if use_learnable_partition:
                torch.nn.utils.clip_grad_value_(log_partition, clip_value)
        optimizer.step()
        if logger is not None:
            logger.log(
                step=epoch * num_steps + step,
                epoch=epoch,
                loss=step_loss.item(),
            )


def evaluate(
    model: torch.nn.Module,
    samples: Optional[torch.Tensor] = None,
    *,
    epoch: Optional[int] = None,
    logger: Optional[manifold.loggers.Logger] = None,
) -> torch.Tensor:
    """Evaluates the model one epoch, computing embeddings in low dimensional space."""
    model.eval()
    # parametric model
    if model.parametric:
        with torch.no_grad():
            # expect samples has the same dtype, device as model
            Y = model(samples).cpu()
    # non-parametric model
    else:
        Y = model.weight.detach().cpu()
    if logger is not None:
        # logger.snapshot(epoch=epoch, embedding=Y.numpy())
        logger.snapshot(epoch=epoch, embedding=Y.clone().numpy())
    return Y


@vtimeit
def fit(
    model: torch.nn.Module,
    criterion: manifold.metrics.Metric,
    optimizer: manifold.optim.Optimizer,
    lr_scheduler: manifold.optim.LRScheduler,
    dataloader: manifold.samplers.DataLoader,
    sampler: manifold.samplers.Sampler,
    *,
    force_resampling: bool = False,
    metric: str = "euclidean",
    clip_grad: bool = True,
    clip_value: float = 4.0,
    use_learnable_partition: bool = False,
    log_partition: Optional[torch.Tensor] = None,
    num_epochs: int = 100,
    snapshot: bool = False,
    snapshot_every: int = 10,
    samples: Optional[torch.Tensor] = None,
    logger: Optional[manifold.loggers.Logger] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Fits the model."""
    vprint(verbose, "fitting model")
    for epoch in range(num_epochs):
        train(
            model,
            criterion,
            optimizer,
            dataloader,
            sampler,
            force_resampling=force_resampling,
            metric=metric,
            clip_grad=clip_grad,
            clip_value=clip_value,
            use_learnable_partition=use_learnable_partition,
            log_partition=log_partition,
            epoch=epoch,
            logger=logger,
        )
        if verbose and isinstance(logger, manifold.loggers.DataFrameLogger):
            epoch_loss = logger.loss(epoch)
            print(f"EPOCH {epoch} loss: {epoch_loss}")
        # snapshot the intermediate states and final state
        if snapshot and logger is not None:
            if (epoch + 1) % snapshot_every == 0 or epoch == num_epochs - 1:
                Y = evaluate(model, samples, epoch=epoch, logger=logger)
        lr_scheduler.step()
    if not snapshot or logger is None:
        Y = evaluate(model, samples)
    return Y


@vtimeit
def transform(
    X: torch.Tensor,
    model: torch.nn.Module,
    verbose: bool = False,
) -> torch.Tensor:
    """Transforms new data."""
    vprint(verbose, "transforming new data")
    model.eval()
    # parametric model
    if model.parametric:
        encoder = model[1]
        with torch.no_grad():
            # expect X has the same dtype, device as the model
            Y = encoder(X).cpu()
    # non-parametric model
    else:
        # TODO
        raise NotImplementedError(
            "'transform' is not implemented for non-parametric models"
        )
    return Y


@vtimeit
def tscne(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    num_components: int = 2,
    *,
    transform: Union[str, Sequence[str], Callable] = "pca",
    max_features: int = 100,
    num_neighbors: int = 10,
    algorithm: str = "annoy",
    metric: str = "euclidean",
    metric_kwargs: dict = {},
    scale_nearest_neighbors: bool = True,
    scaling_start: int = 3,
    scaling_end: int = 6,
    num_extra_neighbors: int = 50,
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
    criterion: str = "infonce",
    criterion_kwargs: dict = {
        "reduction": "sum",
        "with_exp": False,
        "with_log": False,
    },
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
    snapshot: bool = False,
    snapshot_every: int = 10,
    logger: str = "dataframe",
    logger_kwargs: dict = {},
    autosave: bool = False,
    root: str = "./runs/",
    nearest_neighbors: Optional[str] = None,
    neighboring_samples: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    trained_model: Optional[str] = None,
    seed: Optional[Union[int, torch.Generator]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
    tag: str = "tscne",
    dataset: str = "mnist",
    verbose: bool = False,
) -> torch.Tensor:
    """t-distributed Stochastic Contrastive Neighbor Embedding (t-SCNE)."""
    # dtype
    dtype = to_dtype(dtype)
    # device
    device = to_device(device)
    vprint(
        verbose,
        f"running {tag} with '{dtype}' floating point on '{device}' device",
    )
    # generator
    generator = to_generator(seed, device=device)
    # logger
    logger = manifold.loggers.logger(
        logger,
        **logger_kwargs,
    )
    # X
    X = X.to(dtype=dtype)
    X, applied_pca = preprocess(
        X,
        transform=transform,
        max_features=max_features,
        verbose=verbose,
    )
    # y
    if y is not None:
        y = y.to(dtype=torch.int64)
    # handle autosave
    os.makedirs(root, exist_ok=True)
    dataset = f"{dataset}_{len(X)}"
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
        num_extra_neighbors=num_extra_neighbors,
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
        dataset=dataset,
        root=root,
        nearest_neighbors=nearest_neighbors,
        neighboring_samples=neighboring_samples,
        generator=generator,
        device=device,
        verbose=verbose,
    )
    # samples
    anchors = samples(X, device=device) if parametric else None
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
        dtype=dtype,
        device=device,
        snapshot=snapshot,
        samples=anchors,
        logger=logger,
        verbose=verbose,
    )
    # return model, evaluate(model=model, samples=anchors), logger
    # log partition
    log_Z = log_partition(
        partition,
        learnable=use_learnable_partition,
        dtype=X.dtype,
        device=device,
    )
    # criterion
    criterion = manifold.metrics.criterion(
        criterion,
        **criterion_kwargs,
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
    # fitting
    Y = fit(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        dataloader,
        sampler,
        force_resampling=force_resampling,
        metric=metric,
        clip_grad=clip_grad,
        clip_value=clip_value,
        use_learnable_partition=use_learnable_partition,
        log_partition=log_Z,
        num_epochs=num_epochs,
        snapshot=snapshot,
        snapshot_every=snapshot_every,
        samples=anchors,
        logger=logger,
        verbose=verbose,
    )
    if autosave:
        comment = "parametric" if parametric else "non-parametric"
        if trained_model is None:
            trained_model = f"{dataset}_{tag}_{comment}_{timestamp()}.pt"
        save(
            model,
            trained_model,
            root=root,
            verbose=verbose,
        )
    return Y, logger


# class interface


class TSCNE(Estimator):
    """t-distributed Stochastic Contrastive Neighbor Embedding (t-SCNE)."""

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
            f"call method '{type(self).__name__}.fit' instead"
        )

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> "TSCNE":
        raise NotImplementedError
        # return self

    def fit_transform(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return tscne(
            X,
            y,
            num_components=self.num_components,
            **self.kwargs,
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"call method '{type(self).__name__}.fit_transform' instead"
        )

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError
