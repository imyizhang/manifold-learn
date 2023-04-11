import torch

# functional interface


def dataset():
    return


# torch.nn.Module interface


class Mammoth(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()


class MNIST(torch.utils.data.Dataset):

    def __init__(self) -> None:
        pass


# https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
class FastTensorDataLoader:

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.n_samples = self.tensors[0].shape[0]
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
            batch = tuple(
                torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.index:self.index + self.batch_size]
                          for t in self.tensors)
        self.index += self.batch_size
        return batch
