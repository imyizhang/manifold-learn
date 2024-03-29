import math
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import torch

from manifold.decorators import timeit

__all__ = (
    "sample",
    "neighbor",
    "neighbor_loader",
    "get_neighbor_loader",
    "neighbor_sampler",
    "get_neighbor_sampler",
    "batch_neighbor_sampler",
    "get_batch_neighbor_sampler",
    "DataLoader",
    "Sampler",
)


# functional interface


@timeit
def random_permutation(
    n: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns a 1D tensor containing a random permutation of integers from `0`
    to `n - 1`.
    """
    # fixme: speed up permutation
    return torch.randperm(
        n,
        # expect device for generator to be desired device
        generator=generator,
        dtype=torch.int64,
        device=device,
    )


@timeit
def combinations(
    input: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Returns a 2D tensor where each row contains `num_samples` indices sampled
    from all the combinations of length `num_samples` of the corresponding row
    of tensor `input` without replacement.
    """
    # fixme: speed up conmbinations
    # expect input to be a 2D tensor
    if num_samples == 1:
        # return input.view(1, -1).T
        return input.reshape(1, -1).T
    if num_samples == input.shape[1]:
        return input
    output = []
    for i in input:
        output.append(
            torch.combinations(i, num_samples, with_replacement=False)
        )
    return torch.vstack(output)


def num_combinations(n: int, k: int) -> int:
    """Returns the number of combinations of length `k` of `n` samples without
    replacement.
    """
    if k == 1:
        return n
    if k == n:
        return 1
    # expect 1 < k < n
    return math.comb(n, k)


@timeit
def multinomial(
    input: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Returns a 2D tensor where each row contains `num_samples` indices sampled
    from the multinomial probability distribution located in the corresponding
    row of tensor `input`.
    """
    # fixme: speed up multinomial
    return torch.multinomial(
        # expect input to be a 2D tensor
        input.to(dtype=torch.float64),
        num_samples,
        replacement,
        # expect device for generator to be the same as input
        generator=generator,
    )


@timeit
def sample(
    n: int,
    size: int,
    num_samples: int,
    safe_n: int = 10000,
    safe_size: int = 1024,
    rejection: Optional[torch.Tensor] = None,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns a 2D tensor where each row contains `num_samples` indices sampled
    from `0` to `n - 1`.
    """
    if rejection is None and replacement:
        return torch.randint(
            n,
            (size, num_samples),
            dtype=torch.int64,
            generator=generator,
            device=device,
        )
    # optimize memory usage, becoming high CPU usage, or CPU-bound
    if n > safe_n and size > safe_size:
        samples = []
        reminder = size % safe_size
        last_batch_size = reminder if reminder > 0 else safe_size
        for i in range(0, size, safe_size):
            # handle last batch whatever it is complete or not
            if i == size - last_batch_size:
                batch_size = last_batch_size
            else:
                batch_size = safe_size
            samples.append(
                multinomial(
                    # expect device for mask to be desired device
                    _sampling_mask(
                        n,
                        batch_size,
                        rejection=rejection[i : i + batch_size],
                        device=device,
                    ),
                    num_samples,
                    replacement=replacement,
                    # expect device for generator to be desired device
                    generator=generator,
                )
            )
        return torch.vstack(samples)
    return multinomial(
        # expect device for mask to be desired device
        _sampling_mask(
            n,
            size,
            rejection=rejection,
            device=device,
        ),
        num_samples,
        replacement=replacement,
        # expect device for generator to be desired device
        generator=generator,
    )


def _sampling_mask(
    n: int,
    size: int,
    rejection: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # fixme: high memory usage, or memory-bound for large n and size
    mask = torch.ones(
        size,
        n,
        dtype=torch.bool,
        device=device,
    )
    if rejection is not None:
        return mask.scatter(
            dim=1,
            index=rejection.to(device=mask.device),
            value=False,
        )
    return mask


def repeat(samples: torch.Tensor, repeats: int) -> torch.Tensor:
    return samples.expand(-1, repeats).reshape(-1, 1)


def neighbor(samples: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.hstack(samples)


def _num_positive_samples(
    num_positive_samples: int,
    max_positive_samples: int,
) -> int:
    # expect num_positive_samples to be a positive integer
    return min(num_positive_samples, max_positive_samples)


def _max_positive_samples(num_neighbors: int) -> int:
    return num_neighbors


def _num_negative_samples(
    num_negative_samples: int,
    max_negative_samples: int,
) -> int:
    # expect num_negative_samples to be a positive integer
    return min(num_negative_samples, max_negative_samples)


def _max_negative_samples(
    num_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    num_positive_samples: Optional[int] = None,
    exclude_neighbors: bool = False,
    num_neighbors: Optional[int] = None,
) -> int:
    # only excluding positive samples
    if (not exclude_anchor_samples) and exclude_positive_samples:
        raise NotImplementedError(
            "the case where only positive samples are excluded for negative sampling is not supported"
        )
    # including all samples
    if (not exclude_anchor_samples) and (not exclude_positive_samples):
        return num_samples
    # only excluding anchor samples
    if exclude_anchor_samples and (not exclude_positive_samples):
        return num_samples - 1
    # excluding anchor samples and all chosen positive samples
    if not exclude_neighbors:
        return num_samples - 1 - num_positive_samples
    # excluding anchor samples and all candidates for positive samples
    return num_samples - 1 - num_neighbors


def _excluded_indices(
    exclude_anchor_samples: bool = False,
    anchor_samples: Optional[torch.Tensor] = None,
    exclude_positive_samples: bool = False,
    positive_samples: Optional[torch.Tensor] = None,
    exclude_neighbors: bool = False,
    neighbor_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    # only excluding positive samples
    if (not exclude_anchor_samples) and exclude_positive_samples:
        raise NotImplementedError(
            "the case where only positive samples are excluded for negative sampling is not supported"
        )
    # including all samples
    if (not exclude_anchor_samples) and (not exclude_positive_samples):
        return None
    # only excluding anchor samples
    if exclude_anchor_samples and (not exclude_positive_samples):
        # expect anchor_samples to be a torch.int64 tensor
        return anchor_samples
    # excluding anchor samples and all chosen positive samples
    if not exclude_neighbors:
        # expect anchor_samples and positive_samples to be torch.int64 tensors with the same device, length
        return neighbor((anchor_samples, positive_samples))
    # excluding anchor samples and all candidates for positive samples
    # expect anchor_samples and neighbor_indices to be torch.int64 tensors with the same device, length
    return neighbor((anchor_samples, neighbor_indices))


def _negative_sampling_mask(
    batch_size: int,
    num_samples: int,
    exclude_anchor_samples: bool = False,
    anchor_samples: Optional[torch.Tensor] = None,
    exclude_positive_samples: bool = False,
    positive_samples: Optional[torch.Tensor] = None,
    exclude_neighbors: bool = False,
    neighbor_indices: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # only excluding positive samples
    if (not exclude_anchor_samples) and exclude_positive_samples:
        raise NotImplementedError(
            "the case where only positive samples are excluded for negative sampling is not supported"
        )
    mask = torch.ones(
        batch_size,
        num_samples,
        dtype=torch.bool,
        device=device,
    )
    # including all samples
    if (not exclude_anchor_samples) and (not exclude_positive_samples):
        return mask
    # only excluding anchor samples
    if exclude_anchor_samples and (not exclude_positive_samples):
        # if anchor_samples.device != mask.device:
        #     raise RuntimeError(
        #         f"expected a '{mask.device}' device type for anchor_samples but found '{anchor_samples.device}'"
        #     )
        return mask.scatter(
            dim=1,
            index=anchor_samples.to(device=mask.device),
            value=False,
        )
    # excluding anchor samples and all chosen positive samples
    if not exclude_neighbors:
        # if anchor_samples.device != mask.device:
        #     raise RuntimeError(
        #         f"expected a '{mask.device}' device type for anchor_samples but found '{anchor_samples.device}'"
        #     )
        # if positive_samples.device != mask.device:
        #     raise RuntimeError(
        #         f"expected a '{mask.device}' device type for positive_samples but found '{positive_samples.device}'"
        #     )
        return mask.scatter(
            dim=1,
            index=neighbor(
                (
                    anchor_samples.to(device=mask.device),
                    positive_samples.to(device=mask.device),
                )
            ),
            value=False,
        )
    # excluding anchor samples and all candidates for positive samples
    # if anchor_samples.device != mask.device:
    #     raise RuntimeError(
    #         f"expected a '{mask.device}' device type for anchor_samples but found '{anchor_samples.device}'"
    #     )
    # if neighbor_indices.device != mask.device:
    #     raise RuntimeError(
    #         f"expected a '{mask.device}' device type for neighbor_indices but found '{neighbor_indices.device}'"
    #     )
    return mask.scatter(
        dim=1,
        index=neighbor(
            (
                anchor_samples.to(device=mask.device),
                neighbor_indices.to(device=mask.device),
            )
        ),
        value=False,
    )


def _neighbor_indices(
    neighbor_indices: torch.Tensor,
    comb_positive_samples: int,
) -> torch.Tensor:
    return repeat(neighbor_indices, comb_positive_samples)


def _neighbor_sampling_mask(
    batch_size: int,
    num_positive_samples: int,
    num_negative_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    mask = torch.ones(
        (batch_size, num_positive_samples + num_negative_samples),
        dtype=torch.bool,
        device=device,
    )
    mask[:, num_positive_samples:] = False
    return mask


def anchor_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns anchor samples for which positive samples will be drawn from nearest neighbors.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        device (torch.device, optional): Device for the tensor of anchor samples. Defaults to None, i.e., the same device as nearest neighbors.

    Returns:
        torch.Tensor: Indices of anchor samples.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, 1), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    return _anchor_samples(
        num_samples,
        _comb_positive_samples(
            num_neighbors,
            _num_positive_samples(
                num_positive_samples,
                _max_positive_samples(num_neighbors),
            ),
        ),
        # force using desired device
        neighbor_indices.device if device is None else device,
    )


def _anchor_samples(
    num_samples: int,
    comb_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    samples = torch.arange(
        num_samples,
        dtype=torch.int64,
        device=device,
    ).view(-1, 1)
    return repeat(samples, comb_positive_samples)


def max_positive_samples(
    neighbor_indices: torch.Tensor,
    /,
) -> int:
    """Returns the maximum number of positive samples can be drawn for each anchor sample from nearest neighbors without replacement and without order.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.

    Returns:
        int: Maximum number of positive samples can be drawn.
    """
    _, num_neighbors = neighbor_indices.shape
    return _max_positive_samples(num_neighbors)


def num_positive_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples_: int,
) -> int:
    """Returns the number of positive samples to draw for each anchor sample from nearest neighbors without replacement and without order, upper bounded by the number of nearest neighbors.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples_ (int, optional): Number of positive samples to draw for each anchor sample. Defaults to 1.

    Returns:
        int: Number of positive samples to draw.
    """
    _, num_neighbors = neighbor_indices.shape
    return _num_positive_samples(
        num_positive_samples_,
        _max_positive_samples(num_neighbors),
    )


def comb_positive_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
) -> int:
    """Returns the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.

    Returns:
        int: Number of combinations to draw positive samples.
    """
    _, num_neighbors = neighbor_indices.shape
    return _comb_positive_samples(
        num_neighbors,
        _num_positive_samples(
            num_positive_samples,
            _max_positive_samples(num_neighbors),
        ),
    )


def _comb_positive_samples(
    num_neighbors: int,
    num_positive_samples: int,
) -> int:
    return num_combinations(num_neighbors, num_positive_samples)


def positive_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws positive samples for each anchor sample from nearest neighbors without replacement and without order.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        device (torch.device, optional): Device for the tensor of positive samples. Defaults to None, i.e, the same device as nearest neighbors.

    Returns:
        torch.Tensor: Indices of positive samples.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, n_positives), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order, and n_positives is the number of positive samples to draw for each anchor sample.
    """
    _, num_neighbors = neighbor_indices.shape
    return _positive_samples(
        neighbor_indices,
        _num_positive_samples(
            num_positive_samples,
            _max_positive_samples(num_neighbors),
        ),
        # force using desired device
        neighbor_indices.device if device is None else device,
    )


def _positive_samples(
    neighbor_indices: torch.Tensor,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    samples = combinations(neighbor_indices, num_positive_samples)
    if device is None or device == samples.device:
        return samples
    return samples.to(device=device)


def max_negative_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the maximum number of negative samples can be drawn for each anchor sample from all samples without replacement and without order.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.

    Returns:
        int: Maximum number of negative samples can be drawn.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    return _max_negative_samples(
        num_samples,
        exclude_anchor_samples,
        exclude_positive_samples,
        _num_positive_samples(
            num_positive_samples,
            _max_positive_samples(num_neighbors),
        ),
        exclude_neighbors,
        num_neighbors,
    )


def num_negative_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    num_negative_samples_: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the number of negative samples to draw for each anchor sample from all samples, upper bounded by the total number of samples.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        num_negative_samples_ (int): Number of negative samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.

    Returns:
        int: Number of negative samples to draw.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    return _num_negative_samples(
        num_negative_samples_,
        _max_negative_samples(
            num_samples,
            exclude_anchor_samples,
            exclude_positive_samples,
            _num_positive_samples(
                num_positive_samples,
                _max_positive_samples(num_neighbors),
            ),
            exclude_neighbors,
            num_neighbors,
        ),
    )


def negative_sampling_mask(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates a mask for negative sampling given nearest neighbors.

    Args:
        neighbor_indices (torch.Tensor): Indices of the nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.
        device (Optional[torch.device], optional): Device for the tensor of the mask. Defaults to None, i.e, the same device as nearest neighbors.

    Returns:
        torch.Tensor: Mask for negative sampling.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, n), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    num_positive_samples = _num_positive_samples(
        num_positive_samples,
        _max_positive_samples(num_neighbors),
    )
    comb_positive_samples = _comb_positive_samples(
        num_neighbors,
        num_positive_samples,
    )
    return _negative_sampling_mask(
        comb_positive_samples * num_samples,
        num_samples,
        exclude_anchor_samples,
        _anchor_samples(
            num_samples,
            comb_positive_samples,
            neighbor_indices.device,
        ),
        exclude_positive_samples,
        _positive_samples(
            neighbor_indices,
            num_positive_samples,
        ),
        exclude_neighbors,
        _neighbor_indices(
            neighbor_indices,
            comb_positive_samples,
        ),
        # force using desired device
        neighbor_indices.device if device is None else device,
    )


def negative_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws negative samples for each anchor sample from all samples.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of negative samples. Defaults to None, i.e, the same device as nearest neighbors.

    Returns:
        torch.Tensor: Indices of negative samples.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, n_negatives), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order, and n_negatives is the number of negative samples to draw for each anchor sample.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    num_positive_samples = _num_positive_samples(
        num_positive_samples,
        _max_positive_samples(num_neighbors),
    )
    comb_positive_samples = _comb_positive_samples(
        num_neighbors,
        num_positive_samples,
    )
    return multinomial(
        _negative_sampling_mask(
            comb_positive_samples * num_samples,
            num_samples,
            exclude_anchor_samples,
            _anchor_samples(
                num_samples,
                comb_positive_samples,
                neighbor_indices.device,
            ),
            exclude_positive_samples,
            _positive_samples(
                neighbor_indices,
                num_positive_samples,
            ),
            exclude_neighbors,
            _neighbor_indices(
                neighbor_indices,
                comb_positive_samples,
            ),
            # force using desired device
            neighbor_indices.device if device is None else device,
        ),
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                num_samples,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
                exclude_neighbors,
                num_neighbors,
            ),
        ),
        replacement=replacement,
        # expect device for generator to be desired device
        generator=generator,
    )


def neighbor_sampling_mask(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates a mask for neighbor sampling.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.
        device (torch.device, optional): Device for the tensor of neighbor samples. Defaults to None, i.e, the same device as nearest neighbors.

    Returns:
        torch.Tensor: Generated mask for neighbor sampling.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, n_positives + n_negatives), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order, n_positives is the number of positive samples to draw for each anchor sample, and n_negatives is the number of negative samples to draw for each anchor sample.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    num_positive_samples = _num_positive_samples(
        num_positive_samples,
        _max_positive_samples(num_neighbors),
    )
    return _neighbor_sampling_mask(
        _comb_positive_samples(
            num_neighbors,
            num_positive_samples,
        )
        * num_samples,
        num_positive_samples,
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                num_samples,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
                exclude_neighbors,
                num_neighbors,
            ),
        ),
        # force using desired device
        neighbor_indices.device if device is None else device,
    )


def neighbor_samples(
    neighbor_indices: torch.Tensor,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws neighbor samples for each anchor sample.

    Args:
        neighbor_indices (torch.Tensor): Indices of nearest neighbors.
        num_positive_samples (int): Number of positive samples to draw for each anchor sample.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude positive samples for negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of neighbor samples. Defaults to None, i.e, the same device as nearest neighbors.

    Returns:
        torch.Tensor: Indices of neighbor samples.

    Shape:
        - Input: (n, n_neighbors), where n is the total number of samples and n_neighbors is the number of nearest neighbors.
        - Output: (c_positives * n, n_positives + n_negatives), where c_positives is the number of combinations to draw positive samples for each anchor sample from nearest neighbors without replacement and without order, n_positives is the number of positive samples to draw for each anchor sample, and n_negatives is the number of negative samples to draw for each anchor sample.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    num_positive_samples = _num_positive_samples(
        num_positive_samples,
        _max_positive_samples(num_neighbors),
    )
    comb_positive_samples = _comb_positive_samples(
        num_neighbors,
        num_positive_samples,
    )
    positve_samples = _positive_samples(
        neighbor_indices,
        num_positive_samples,
        # force using desired device
        neighbor_indices.device if device is None else device,
    )
    return neighbor(
        (
            positve_samples,
            multinomial(
                _negative_sampling_mask(
                    comb_positive_samples * num_samples,
                    num_samples,
                    exclude_anchor_samples,
                    _anchor_samples(
                        num_samples,
                        comb_positive_samples,
                        neighbor_indices.device,
                    ),
                    exclude_positive_samples,
                    positve_samples,
                    exclude_neighbors,
                    _neighbor_indices(
                        neighbor_indices,
                        comb_positive_samples,
                    ),
                    # force using desired device
                    neighbor_indices.device if device is None else device,
                ),
                _num_negative_samples(
                    num_negative_samples,
                    _max_negative_samples(
                        num_samples,
                        exclude_anchor_samples,
                        exclude_positive_samples,
                        num_positive_samples,
                        exclude_neighbors,
                        num_neighbors,
                    ),
                ),
                replacement=replacement,
                # expect device for generator to be desired device
                generator=generator,
            ),
        )
    )


def batch_anchor_samples(
    neighboring_samples: torch.Tensor,
    /,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns anchor samples in a batch.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        device (torch.device, optional): Device for the tensor of anchor samples. Defaults to None, i.e, the same device as neighboring samples.

    Returns:
        torch.Tensor: Indices of anchor samples.

    Shape:
        - Input: (B, 1 + n_positives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
        - Output: (B, 1)
    """
    return _batch_anchor_samples(
        neighboring_samples,
        # force using desired device
        device,
    )


def _batch_anchor_samples(
    neighboring_samples: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    samples = neighboring_samples[:, :1]
    if device is None or device == samples.device:
        return samples
    return samples.to(device=device)


def batch_positive_samples(
    neighboring_samples: torch.Tensor,
    /,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns chosen positive samples in a batch.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        device (torch.device, optional): Device for the tensor of positive samples. Defaults to None, i.e, the same device as neighboring samples.

    Returns:
        torch.Tensor: Indices of chosen positive samples.

    Shape:
        - Input: (B, 1 + n_positives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
        - Output: (B, n_positives)
    """
    return _batch_positive_samples(
        neighboring_samples,
        num_positive_samples,
        # force using desired device
        device,
    )


def _batch_positive_samples(
    neighboring_samples: torch.Tensor,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    samples = neighboring_samples[:, 1 : 1 + num_positive_samples]
    if device is None or device == samples.device:
        return samples
    return samples.to(device=device)


def batch_max_negative_samples(
    num_samples: int,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the maximum number of negative samples can be drawn for each anchor sample in a batch from all samples without replacement and without order.

    Args:
        num_samples (int): Total number of samples.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.

    Returns:
        int: Maximum number of negative samples can be drawn.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    return _max_negative_samples(
        num_samples,
        exclude_anchor_samples,
        exclude_positive_samples,
        num_positive_samples,
    )


def batch_num_negative_samples(
    num_samples: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the number of negative samples to draw for each anchor sample in a batch from all samples, upper bounded by the total number of samples.

    Args:
        num_samples (int): Total number of samples.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.

    Returns:
        int: Number of negative samples to draw.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    return _num_negative_samples(
        num_negative_samples,
        _max_negative_samples(
            num_samples,
            exclude_anchor_samples,
            exclude_positive_samples,
            num_positive_samples,
        ),
    )


def batch_negative_sampling_mask(
    neighboring_samples: torch.Tensor,
    num_samples: int,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates a mask for negative sampling given a batch of neighboring samples.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        num_samples (int): Total number of samples.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.
        device (torch.device, optional): Device for the tensor of mask. Defaults to None, i.e, the same device as neighboring samples.

    Returns:
        torch.Tensor: Generated mask for batch negative sampling.

    Shape:
        - Input: (B, 1 + n_positives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
        - Output: (B, n), where n is the total number of samples.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    batch_size, _ = neighboring_samples.shape
    return _negative_sampling_mask(
        batch_size,
        num_samples,
        exclude_anchor_samples,
        neighboring_samples[:, :1],
        exclude_positive_samples,
        neighboring_samples[:, 1 : 1 + num_positive_samples],
        # force using desired device
        neighboring_samples.device if device is None else device,
    )


def batch_negative_samples(
    neighboring_samples: torch.Tensor,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    num_samples: Optional[int] = None,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws negative samples for each anchor sample in a batch from all samples.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        num_samples (int, optional): Total number of samples. Defaults to None, i.e., batch negative sampling can be skipped.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of negative samples. Defaults to None, i.e, the same device as neighboring samples.

    Returns:
        torch.Tensor: Indices of negative samples.

    Shape:
        - Input: (B, 1 + n_positives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
        - Output: (B, n_negatives), where n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if num_samples is None:
        return _batch_negative_samples(
            neighboring_samples,
            num_positive_samples,
            num_negative_samples,
            # force using desired device
            device,
        )
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    batch_size, _ = neighboring_samples.shape
    return multinomial(
        _negative_sampling_mask(
            batch_size,
            num_samples,
            exclude_anchor_samples,
            neighboring_samples[:, :1],
            exclude_positive_samples,
            neighboring_samples[:, 1 : 1 + num_positive_samples],
            # force using desired device
            neighboring_samples.device if device is None else device,
        ),
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                num_samples,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
            ),
        ),
        replacement=replacement,
        # expect device for generator to be desired device
        generator=generator,
    )


def _batch_negative_samples(
    neighboring_samples: torch.Tensor,
    num_positive_samples: int,
    num_negative_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    start = 1 + num_positive_samples
    samples = neighboring_samples[:, start : start + num_negative_samples]
    if device is None or device == samples.device:
        return samples
    return samples.to(device=device)


def batch_neighbor_sampling_mask(
    neighboring_samples: torch.Tensor,
    num_samples: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates a mask for neighbor sampling in a batch.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        num_samples (int): Total number of samples.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.
        device (torch.device, optional): Device for the tensor of mask. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Generated mask for neighbor sampling in the batch.

    Shape:
        - Output: (B, n_positives + n_negatives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch, and n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    batch_size, _ = neighboring_samples.shape
    return _neighbor_sampling_mask(
        batch_size,
        num_positive_samples,
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                num_samples,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
            ),
        ),
        # force using desired device
        neighboring_samples.device if device is None else device,
    )


def batch_neighbor_samples(
    neighboring_samples: torch.Tensor,
    num_samples: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws neighbor samples for each anchor sample in a batch.

    Args:
        neighboring_samples (torch.Tensor): Indices of neighboring samples in the batch.
        num_samples (int): Total number of samples.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for batch negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of neighbor samples. Defaults to None, i.e, the same device as neighboring samples.

    Returns:
        torch.Tensor: Indices of neighbor samples.

    Shape:
        - Input: (B, 1 + n_positives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
        - Output: (B, n_positives + n_negatives), where n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for batch negative sampling is not supported"
        )
    batch_size, _ = neighboring_samples.shape
    return neighbor(
        (
            _batch_positive_samples(
                neighboring_samples,
                num_positive_samples,
                device,
            ),
            multinomial(
                _negative_sampling_mask(
                    batch_size,
                    num_samples,
                    exclude_anchor_samples,
                    neighboring_samples[:, :1],
                    exclude_positive_samples,
                    neighboring_samples[:, 1 : 1 + num_positive_samples],
                    # force using desired device
                    neighboring_samples.device if device is None else device,
                ),
                _num_negative_samples(
                    num_negative_samples,
                    _max_negative_samples(
                        num_samples,
                        exclude_anchor_samples,
                        exclude_positive_samples,
                        num_positive_samples,
                    ),
                ),
                replacement=replacement,
                # expect device for generator to be desired device
                generator=generator,
            ),
        )
    )


def in_batch_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Stacks embeddings of neighboring samples in a batch vertically.

    Args:
        embeddings (torch.Tensor): Embeddings of neighboring samples.

    Returns:
        torch.Tensor: Embeddings of neighboring samples stacked vertically.

    Shape:
        - Input: (B, 1 + n_positives, d), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch, and d is the embedding dimension.
        - Output: ((1 + n_positives) * B, d)
    """
    # if embeddings.size(dim=-2) == 2:
    #   return torch.vstack((embeddings[:, 0, :], embeddings[:, 1, :]))
    return embeddings.T.reshape(embeddings.size(dim=-1), -1).T


def in_batch_anchor_samples(
    batch_size: int,
    /,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns anchor samples whose indices come from a batch.

    Args:
        batch_size (int): Batch size.
        device (torch.device, optional): Device for the tensor of anchor samples. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Indices of anchor samples.

    Shape:
        - Output: (B, 1), where B is the batch size.
    """
    return _in_batch_anchor_samples(
        batch_size,
        # force using desired device
        device,
    )


def _in_batch_anchor_samples(
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.arange(
        batch_size,
        dtype=torch.int64,
        device=device,
    ).view(-1, 1)


def in_batch_positive_samples(
    batch_size: int,
    /,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns positive samples whose indices come from a batch.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        device (torch.device, optional): Device for the tensor of positive samples. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Indices of positive samples.

    Shape:
        - Output: (n_positives * B, 1), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
    """
    return _in_batch_positive_samples(
        batch_size,
        num_positive_samples,
        # force using desired device
        device,
    )


def _in_batch_positive_samples(
    batch_size: int,
    num_positive_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return (
        torch.arange(
            batch_size,
            (1 + num_positive_samples) * batch_size,
            dtype=torch.int64,
            device=device,
        )
        .view(num_positive_samples, -1)
        .T
    )


def in_batch_max_negative_samples(
    batch_size: int,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the maximum number of negative samples can be drawn for each anchor sample from a batch without replacement and without order.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.

    Returns:
        int: Maximum number of negative samples can be drawn.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return _max_negative_samples(
        batch_size,
        exclude_anchor_samples,
        exclude_positive_samples,
        num_positive_samples,
    )


def in_batch_num_negative_samples(
    batch_size: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
) -> int:
    """Returns the number of negative samples to draw for each anchor sample from a batch, upper bounded by the batch size.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.

    Returns:
        int: Number of negative samples to draw.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return _num_negative_samples(
        num_negative_samples,
        _max_negative_samples(
            batch_size,
            exclude_anchor_samples,
            exclude_positive_samples,
            num_positive_samples,
        ),
    )


def in_batch_negative_sampling_mask(
    batch_size: int,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns a mask for in-batch negative sampling.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.
        device (torch.device, optional): Device for the tensor of mask. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Generated mask for in-batch negative sampling.

    Shape:
        - Output: (B, (1 + n_positives) * B), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return _in_batch_negative_sampling_mask(
        batch_size,
        num_positive_samples,
        exclude_anchor_samples,
        exclude_positive_samples,
        # force using desired device
        device,
    )


def _in_batch_negative_sampling_mask(
    batch_size: int,
    /,
    num_positive_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # only excluding positive samples
    if (not exclude_anchor_samples) and exclude_positive_samples:
        raise NotImplementedError(
            "the case where only positive samples are excluded for negative sampling is not supported"
        )
    excluding = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    including = torch.ones_like(excluding)
    # including all samples
    if (not exclude_anchor_samples) and (not exclude_positive_samples):
        return torch.hstack(
            1 * [including] + num_positive_samples * [including]
        )
    # only excluding anchor samples
    if exclude_anchor_samples and (not exclude_positive_samples):
        return torch.hstack(
            1 * [excluding] + num_positive_samples * [including]
        )
    # excluding anchor samples and all chosen positive samples
    return torch.hstack(1 * [excluding] + num_positive_samples * [excluding])


def in_batch_negative_samples(
    batch_size: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws negative samples for each anchor sample whose indices come from a batch.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of negative samples. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Indices of negative samples.

    Shape:
        - Output: (B, n_negatives), where B is the batch size, n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return multinomial(
        _in_batch_negative_sampling_mask(
            batch_size,
            num_positive_samples,
            exclude_anchor_samples,
            exclude_positive_samples,
            # force using desired device
            device,
        ),
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                batch_size,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
            ),
        ),
        replacement=replacement,
        # expect device for generator to be desired device
        generator=generator,
    )


def in_batch_neighbor_sampling_mask(
    batch_size: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates a mask for neighbor sampling in a batch.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.
        device (torch.device, optional): Device for the tensor of mask. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Generated mask for neighbor sampling in the batch.

    Shape:
        - Output: (B, n_positives + n_negatives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch, and n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return _neighbor_sampling_mask(
        batch_size,
        num_positive_samples,
        _num_negative_samples(
            num_negative_samples,
            _max_negative_samples(
                batch_size,
                exclude_anchor_samples,
                exclude_positive_samples,
                num_positive_samples,
            ),
        ),
        # force using desired device
        device,
    )


def in_batch_neighbor_samples(
    batch_size: int,
    /,
    num_positive_samples: int,
    num_negative_samples: int,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draws neighbor samples for each anchor sample whose indices come from a batch.

    Args:
        batch_size (int): Batch size.
        num_positive_samples (int): Number of chosen positive samples for each anchor sample in the batch.
        num_negative_samples (int): Number of negative samples to draw for each anchor sample in the batch.
        exclude_anchor_samples (bool, optional): Whether to exclude anchor samples for in-batch negative sampling. Defaults to False.
        exclude_positive_samples (bool, optional): Whether to exclude chosen positive samples for in-batch negative sampling. Defaults to False.
        exclude_neighbors (bool, optional): Whether to exclude nearest neighbors, i.e., all candidates for positive samples, for in-batch negative sampling. Defaults to False.
        replacement (bool, optional): Whether to draw negative samples with replacement. Defaults to False.
        generator (torch.Generator, optional): Generator for random sampling. Defaults to None.
        device (torch.device, optional): Device for the tensor of neighbor samples. Defaults to None, i.e., the current device for the default tensor type (see `torch.set_default_tensor_type()`).

    Returns:
        torch.Tensor: Indices of neighbor samples.

    Shape:
        - Output: (B, n_positives + n_negatives), where B is the batch size, n_positives is the number of chosen positive samples for each anchor sample in the batch, and n_negatives is the number of negative samples to draw for each anchor sample in the batch.
    """
    if exclude_neighbors:
        raise NotImplementedError(
            "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
        )
    return neighbor(
        (
            _in_batch_positive_samples(
                batch_size,
                num_positive_samples,
                # force using desired device
                device,
            ),
            multinomial(
                _in_batch_negative_sampling_mask(
                    batch_size,
                    num_positive_samples,
                    exclude_anchor_samples,
                    exclude_positive_samples,
                    # force using desired device
                    device,
                ),
                _num_negative_samples(
                    num_negative_samples,
                    _max_negative_samples(
                        batch_size,
                        exclude_anchor_samples,
                        exclude_positive_samples,
                        num_positive_samples,
                    ),
                ),
                replacement=replacement,
                # expect device for generator to be desired device
                generator=generator,
            ),
        )
    )


def neighbor_loader(
    neighboring_samples: torch.Tensor,
    /,
    **kwargs,
) -> "NeighborLoader":
    """Returns a `manifold.samplers.NeighborLoader`."""
    return NeighborLoader(neighboring_samples, **kwargs)


get_neighbor_loader = neighbor_loader


def neighbor_sampler(
    num_positive_samples: int,
    num_negative_samples: Optional[int] = None,
    force_resampling: bool = False,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    repeat: int = 1,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> "NeighborSampler":
    """Returns a `manifold.samplers.NeighborSampler`."""
    # negative sampling will be performed in each batch, parameters for negative sampling are thus ignored
    if force_resampling:
        return NeighborSampler(
            num_positive_samples,
            device=device,
        )
    # perform negative sampling first
    return NeighborSampler(
        num_positive_samples,
        num_negative_samples,
        exclude_anchor_samples=exclude_anchor_samples,
        exclude_positive_samples=exclude_positive_samples,
        exclude_neighbors=exclude_neighbors,
        replacement=replacement,
        generator=generator,
        device=device,
    )


get_neighbor_sampler = neighbor_sampler


def batch_neighbor_sampler(
    num_positive_samples: int,
    num_negative_samples: int,
    force_resampling: bool = False,
    in_batch: bool = False,
    exclude_anchor_samples: bool = False,
    exclude_positive_samples: bool = False,
    exclude_neighbors: bool = False,
    replacement: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> "BatchNeighborSampler":
    """Returns a `manifold.samplers.BatchNeighborSampler`."""
    if force_resampling:
        # perform in-batch negative sampling in a batch
        if in_batch:
            return InBatchNeighborSampler(
                num_positive_samples,
                num_negative_samples,
                exclude_anchor_samples=exclude_anchor_samples,
                exclude_positive_samples=exclude_positive_samples,
                exclude_neighbors=exclude_neighbors,
                replacement=replacement,
                generator=generator,
                device=device,
            )
        # perform negative sampling in a batch
        return BatchNeighborSampler(
            num_positive_samples,
            num_negative_samples,
            exclude_anchor_samples=exclude_anchor_samples,
            exclude_positive_samples=exclude_positive_samples,
            exclude_neighbors=exclude_neighbors,
            replacement=replacement,
            generator=generator,
            device=device,
        )
    # perform negative sampling by slicing each batch, parameters for negative sampling are thus ignored
    return BatchNeighborSampler(
        num_positive_samples,
        num_negative_samples,
        device=device,
    )


get_batch_neighbor_sampler = batch_neighbor_sampler


# class interface


T_co = TypeVar("T_co", covariant=True)


class DataLoader(Generic[T_co]):
    """Base class for all DataLoaders."""

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> "DataLoader[T_co]":
        raise NotImplementedError

    def __next__(self) -> T_co:
        raise NotImplementedError


class Sampler(Generic[T_co]):
    """Base class for all Samplers."""

    def __init__(self) -> None:
        pass

    def __call__(self) -> T_co:
        raise NotImplementedError


class NeighborLoader(DataLoader[torch.Tensor]):
    """DataLoader for neighboring samples.

    Reference:
        [1] https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(
        self,
        neighboring_samples: torch.Tensor,
        /,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # expect neighboring_samples to be a 2D, torch.int64 tensor
        super().__init__()
        # expect device for samples to be desired device
        if device is None:
            self.samples = neighboring_samples
            self.device = neighboring_samples.device
        # force using desired device for samples
        else:
            self.samples = neighboring_samples.to(device=device)
            self.device = device
        self.num_samples, _ = neighboring_samples.shape
        # batch gradient descent will be used
        if batch_size is None or batch_size >= self.num_samples:
            self.batch_size = self.num_samples
            self.last_batch_size = self.num_samples
            self.drop_last = False
            self.num_batches = 1
        # mini-batch stochastic gradient descent will be used
        else:
            self.batch_size = batch_size
            self.last_batch_size = batch_size
            self.num_batches, remainder = divmod(self.num_samples, batch_size)
            self.drop_last = drop_last
            if remainder > 0 and not drop_last:
                self.last_batch_size = remainder
                self.num_batches += 1
        self.shuffle = shuffle
        # force using desired device for generator
        if generator is None:
            self.generator = torch.Generator(device)
        # expect device for generator to be desired device
        else:
            self.generator = generator

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> "NeighborLoader[torch.Tensor]":
        # sample batch randomly
        if self.shuffle:
            self._indices = random_permutation(
                self.num_samples,
                # expect device for generator to be desired device
                generator=self.generator,
                device=self.device,
            )
        # sample batch sequentially
        else:
            self._indices = None
        # reset index pointer
        self._index = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self._index >= min(
            self.num_batches * self.batch_size, self.num_samples
        ):
            raise StopIteration
        next_index = self._index + self.batch_size
        # sample batch randomly
        if self._indices is not None:
            batch = self.samples[self._indices[self._index : next_index]]
        # sample batch sequentially
        else:
            batch = self.samples[self._index : next_index]
        # update index pointer
        self._index = next_index
        # expect device for batch to be desired device
        return batch


class NeighborSampler(Sampler[torch.Tensor]):
    """Draws neighboring samples given nearest neighbors."""

    def __init__(
        self,
        num_positive_samples: int,
        num_negative_samples: Optional[int] = None,
        exclude_anchor_samples: bool = False,
        exclude_positive_samples: bool = False,
        exclude_neighbors: bool = False,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # expect num_positive_samples > 0
        # expect num_negative_samples >= 0
        super().__init__()
        self._num_positive_samples = num_positive_samples
        self._num_negative_samples = num_negative_samples
        self.exclude_anchor_samples = exclude_anchor_samples
        self.exclude_positive_samples = exclude_positive_samples
        self.exclude_neighbors = exclude_neighbors
        self.replacement = replacement
        self.generator = generator
        self._device = device

    def __call__(
        self,
        neighbor_indices: torch.Tensor,
        /,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # expect neighbor_indices to be a 2D, torch.int64 tensor
        self._samples = neighbor_indices
        self._labels = labels
        num_samples, num_neighbors = neighbor_indices.shape
        self._max_positive_samples = _max_positive_samples(num_neighbors)
        self._comb_positive_samples = _comb_positive_samples(
            num_neighbors,
            self.num_positive_samples,
        )
        self._anchor_samples = _anchor_samples(
            num_samples,
            self._comb_positive_samples,
            # force using desired device for anchor samples
            self.device,
        )
        self._positive_samples = _positive_samples(
            neighbor_indices,
            self.num_positive_samples,
            # force using desired device for positive samples
            self.device,
        )
        # perform negative sampling first
        if (
            self._num_negative_samples is not None
            and self._num_negative_samples > 0
        ):
            self._max_negative_samples = _max_negative_samples(
                num_samples,
                self.exclude_anchor_samples,
                self.exclude_positive_samples,
                self.num_positive_samples,
                self.exclude_neighbors,
                num_neighbors,
            )
            # fixme: negative sampling mask is not supported
            self._negative_sampling_mask = None
            self._negative_samples = sample(
                num_samples,
                self._comb_positive_samples * num_samples,
                self.num_negative_samples,
                rejection=_excluded_indices(
                    self.exclude_anchor_samples,
                    self._anchor_samples,
                    self.exclude_positive_samples,
                    self._positive_samples,
                    self.exclude_neighbors,
                    _neighbor_indices(
                        neighbor_indices,
                        self._comb_positive_samples,
                    ),
                ),
                replacement=self.replacement,
                generator=self.generator,
                device=self.device,
            )
            self._neighbor_samples = neighbor(
                (self._positive_samples, self._negative_samples)
            )
            if labels is None:
                self._neighbor_sampling_mask = _neighbor_sampling_mask(
                    self._comb_positive_samples * num_samples,
                    self.num_positive_samples,
                    self.num_negative_samples,
                    # force using desired device
                    self.device,
                )
            # fixme: easy but number of positive samples and number of negative samples will be affected
            else:
                # expect labels to be a 1D, torch.int64 tensor
                self._neighbor_sampling_mask = (
                    labels[self._neighbor_samples]
                    == labels[self._anchor_samples]
                )
                self._num_positive_samples = self._neighbor_sampling_mask.sum(
                    dim=1
                )
                print(
                    "number of positives for each anchor: mean {mean}, min {min}, max {max}".format(
                        mean=self._num_positive_samples.mean().cpu().item(),
                        min=self._num_positive_samples.min().cpu().item(),
                        max=self._num_positive_samples.max().cpu().item(),
                    )
                )
                self._num_positive_samples = (
                    self._num_positive_samples.mean().cpu().item()
                )
        # negative sampling will be performed in each batch, parameters for negative sampling are thus ignored
        else:
            if labels is not None:
                raise NotImplementedError(
                    "negative sampling will be performed in each batch, supervised setup is not supported here"
                )
            self._max_negative_samples = None
            self._negative_sampling_mask = None
            self._negative_samples = None
            self._neighbor_samples = self._positive_samples
            self._neighbor_sampling_mask = None
        return neighbor((self._anchor_samples, self._neighbor_samples))

    @property
    def device(self) -> torch.device:
        # expect device for samples to be desired device
        if hasattr(self, "_samples") and self._device is None:
            return self._samples.device
        # force using desired device
        return self._device

    @property
    def max_positive_samples(self) -> int:
        self._validate_attribute("_max_positive_samples")
        return self._max_positive_samples

    @property
    def num_positive_samples(self) -> int:
        if (
            hasattr(self, "_max_positive_samples")
            and self._max_positive_samples is not None
        ):
            return _num_positive_samples(
                self._num_positive_samples,
                self._max_positive_samples,
            )
        else:
            return self._num_positive_samples

    @property
    def anchor_samples(self) -> torch.Tensor:
        self._validate_attribute("_anchor_samples")
        return self._anchor_samples

    @property
    def positive_samples(self) -> torch.Tensor:
        self._validate_attribute("_positive_samples")
        return self._positive_samples

    @property
    def max_negative_samples(self) -> int:
        self._validate_attribute("_max_negative_samples")
        return self._max_negative_samples

    @property
    def num_negative_samples(self) -> int:
        if self._num_negative_samples is not None:
            if (
                hasattr(self, "_max_negative_samples")
                and self._max_negative_samples is not None
            ):
                return _num_negative_samples(
                    self._num_negative_samples,
                    self._max_negative_samples,
                )
            else:
                return self._num_negative_samples
        return 0

    @property
    def negative_sampling_mask(self) -> torch.Tensor:
        self._validate_attribute("_negative_sampling_mask")
        return self._negative_sampling_mask

    @property
    def negative_samples(self) -> torch.Tensor:
        self._validate_attribute("_negative_samples")
        return self._negative_samples

    @property
    def neighbor_sampling_mask(self) -> torch.Tensor:
        self._validate_attribute("_neighbor_sampling_mask")
        return self._neighbor_sampling_mask

    @property
    def neighbor_samples(self) -> torch.Tensor:
        self._validate_attribute("_neighbor_samples")
        return self._neighbor_samples

    def _validate_attribute(self, attribute: str) -> None:
        if not hasattr(self, attribute):
            raise RuntimeError(
                f"call '{type(self).__name__}()' first before accessing attribute '{attribute}'"
            )


class BatchNeighborSampler(NeighborSampler):
    """Draws anchor and neighbor samples for each batch."""

    def __init__(
        self,
        num_positive_samples: int,
        num_negative_samples: int,
        exclude_anchor_samples: bool = False,
        exclude_positive_samples: bool = False,
        exclude_neighbors: bool = False,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            num_positive_samples,
            num_negative_samples,
            exclude_anchor_samples,
            exclude_positive_samples,
            exclude_neighbors,
            replacement,
            generator,
            device,
        )
        self._batch_size = None
        self._embedding = None

    def __call__(
        self,
        neighboring_samples: torch.Tensor,
        num_samples: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        /,
        force_resampling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._samples = neighboring_samples
        self._labels = labels
        self._max_postive_samples = None
        batch_size, _ = neighboring_samples.shape
        self._anchor_samples = _batch_anchor_samples(
            neighboring_samples,
            # force using desired device
            self.device,
        )
        self._positive_samples = _batch_positive_samples(
            neighboring_samples,
            self.num_positive_samples,
            # force using desired device
            self.device,
        )
        # negative sampling
        if num_samples is None or (not force_resampling):
            if force_resampling:
                raise ValueError(
                    f"batch negative sampling is always skipped when calling '{type(self).__name__}()'"
                )
            self._max_negative_samples = None
            self._negative_sampling_mask = None
            self._negative_samples = _batch_negative_samples(
                neighboring_samples,
                self.num_positive_samples,
                self.num_negative_samples,
                # force using desired device
                self.device,
            )
        # have to force resampling as long as dataloader is shuffled, except that negative sampling is already performed in neighbor sampler
        else:
            if self.exclude_neighbors:
                raise NotImplementedError(
                    "the case where nearest neighbors are excluded for batch negative sampling is not supported"
                )
            self._max_negative_samples = _max_negative_samples(
                num_samples,
                self.exclude_anchor_samples,
                self.exclude_positive_samples,
                self.num_positive_samples,
            )
            self._negative_sampling_mask = _negative_sampling_mask(
                batch_size,
                num_samples,
                self.exclude_anchor_samples,
                self._anchor_samples,
                self.exclude_positive_samples,
                self._positive_samples,
                # force using desired device
                self.device,
            )
            self._negative_samples = multinomial(
                self._negative_sampling_mask,
                self.num_negative_samples,
                replacement=self.replacement,
                # expect device for generator to be desired device
                generator=self.generator,
            )
        if labels is None:
            # update batch size dependent only properties
            if batch_size != self._batch_size and labels is None:
                self._batch_size = batch_size
                self._neighbor_sampling_mask = _neighbor_sampling_mask(
                    self._comb_positive_samples * num_samples,
                    self.num_positive_samples,
                    self.num_negative_samples,
                    # force using desired device
                    self.device,
                )
        # fixme: number of positive samples and number of negative samples will be affected
        else:
            # expect labels to be a 1D, torch.int64 tensor
            self._neighbor_sampling_mask = (
                labels[self._neighbor_samples] == labels[self._anchor_samples]
            )
            self._num_positive_samples = self._neighbor_sampling_mask.sum(dim=1)
            print(
                "number of positives for each anchor: mean {mean}, min {min}, max {max}".format(
                    mean=self._num_positive_samples.mean().cpu().item(),
                    min=self._num_positive_samples.min().cpu().item(),
                    max=self._num_positive_samples.max().cpu().item(),
                )
            )
            self._num_positive_samples = (
                self._num_positive_samples.mean().cpu().item()
            )
        self._neighbor_samples = neighbor(
            (self._positive_samples, self._negative_samples)
        )
        return self._anchor_samples, self._neighbor_samples

    @property
    def embedding(self) -> torch.nn.Module:
        return self._embedding

    @embedding.setter
    def embedding(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Embedding):
            raise TypeError(
                f"expected 'torch.nn.Embedding', but got '{type(model)}'"
            )
        # if next(model.parameters()).device != self.device:
        #     raise RuntimeError(
        #         f"expected a '{self.device}' device type for embedding, but got '{next(model.parameters()).device}'"
        #     )
        # expect device for model to be desired device
        self._embedding = model

    def embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            raise RuntimeError(
                "set attribute 'embedding' first before calling method 'embeddings()'"
            )
        # if indices.device != self.device:
        #     raise RuntimeError(
        #         f"expected a '{self.device}' device type for indices, but got '{indices.device}'"
        #     )
        # expect device for indices to be desired device
        return self._embedding(indices)


class InBatchNeighborSampler(BatchNeighborSampler):
    """Draws anchor and neighbor samples for each batch, whose indices come from the batch."""

    def __call__(
        self,
        neighboring_samples: torch.Tensor,
        num_samples: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        /,
        force_resampling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._samples = neighboring_samples
        self._labels = labels
        self._max_positive_samples = None
        batch_size, _ = neighboring_samples.shape
        # update batch size dependent only properties
        sampled = True
        if batch_size != self._batch_size:
            self._batch_size = batch_size
            self._anchor_samples = _in_batch_anchor_samples(
                batch_size,
                # force using desired device
                self.device,
            )
            self._positive_samples = _in_batch_positive_samples(
                batch_size,
                self.num_positive_samples,
                # force using desired device
                self.device,
            )
            # negative sampling
            if self.exclude_neighbors:
                raise NotImplementedError(
                    "the case where nearest neighbors are excluded for in-batch negative sampling is not supported"
                )
            self._max_negative_samples = _max_negative_samples(
                batch_size,
                self.exclude_anchor_samples,
                self.exclude_positive_samples,
                self.num_positive_samples,
            )
            self._negative_sampling_mask = _in_batch_negative_sampling_mask(
                batch_size,
                self.num_positive_samples,
                self.exclude_anchor_samples,
                self.exclude_positive_samples,
                # force using desired device
                self.device,
            )
            if labels is None:
                self._neighboring_sampling_mask = _neighbor_sampling_mask(
                    batch_size,
                    self.num_positive_samples,
                    self.num_negative_samples,
                    # force using desired device
                    self.device,
                )
            sampled = False
        # always implictly force resampling even `force_resampling` is False as long as dataloader is shuffled
        if (not sampled) or force_resampling:
            self._negative_samples = multinomial(
                self._negative_sampling_mask,
                self.num_negative_samples,
                replacement=self.replacement,
                # expect device for generator to be desired device
                generator=self.generator,
            )
        self._neighbor_samples = neighbor(
            (self._positive_samples, self._negative_samples)
        )
        return self._anchor_samples, self._neighbor_samples

    @property
    def embedding(self) -> torch.nn.Module:
        return self._embedding

    @embedding.setter
    def embedding(self, model: torch.nn.Module) -> None:
        if not hasattr(self, "_samples"):
            raise RuntimeError(
                f"call method '{type(self).__name__}()' first before setting attribute 'embedding'"
            )
        if not isinstance(model, torch.nn.Embedding):
            raise TypeError(
                f"expected 'torch.nn.Embedding', but got '{type(model)}'"
            )
        # if next(model.parameters()).device != self.device:
        #     raise RuntimeError(
        #         f"expected a '{self.device}' device type for embedding, but got '{next(model.parameters()).device}'"
        #     )
        # expect device for model to be desired device
        self._embedding = model
        if self._samples.device != self.device:
            # force using desired device
            self._embeddings = model(self._samples.to(self.device))
        else:
            self._embeddings = model(self._samples)
        self._embeddings = in_batch_embeddings(self._embeddings)

    def embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            raise RuntimeError(
                "set attribute 'embedding' first before calling method 'embeddings()'"
            )
        # if indices.device != self.device:
        #     raise RuntimeError(
        #         f"expected a '{self.device}' device type for indices, but got '{indices.device}'"
        #     )
        # expect device for indices to be desired device
        return self._embeddings[indices]
