import torch


def neighbor_loader(
    neighbor_graph: torch.Tensor,
    batch_size: int = 1,
    shuffle: bool = False,
):

    return


# based on https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self,
                 neighbor_mat,
                 batch_size=1024,
                 shuffle=False,
                 on_gpu=False,
                 drop_last=False,
                 seed=0):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [
            torch.tensor(neighbor_mat.row),
            torch.tensor(neighbor_mat.col)
        ]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        self.device = "cpu"
        if on_gpu:
            self.device = "cuda"
            tensors = [tensor.to(self.device) for tensor in tensors]
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

        self.batch_size = torch.tensor(self.batch_size,
                                       dtype=int).to(self.device)

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.dataset_len - self.batch_size:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = tuple(
                torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(
                t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches