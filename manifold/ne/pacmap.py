from typing import Optional, Tuple

import torch

from neighbors import NearestNeighbors
from distances import PairwiseDistance
from .base import Estimator


def _neighbor_graph(X):
    return


def _NN_samples(n_):
    return


def _MN_samples():
    return


def _FP_samples():
    return


def _similarity_graph(X, binary: bool = False):
    return


def _similarity_graph_embedded(graph_embeded):
    return


def _loss(P, Q):
    return


def _weight(itr):
    if itr < 100:
        w_NN = 2.
        w_MN = (1 - itr / 100) * 1000. + itr / 100 * 3.
        w_FP = 1.
    elif itr < 200:
        w_NN = 3.
        w_MN = 3.
        w_FP = 1.
    else:
        w_NN = 1.
        w_MN = 0.
        w_FP = 1.
    return w_NN, w_MN, w_FP


def _gradient_descent(loss):
    return


def pacmap(
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    n_components: int = 2,
    *,
    init: str = 'pca',
    n_iters: int = 1000,
    lr: float = 10e-3,
    random_state: Optional[int] = None,
):
    '''PaCMAP optimization for new data.
    '''
    m, _ = X.shape[-2:]

    if intermediate:
        intermediate_states = np.empty((len(inter_snapshots), n, n_components),
                                       dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if init == 'pca':
        if pca_solution:
            Y = np.concatenate([embedding, 0.01 * X[:, :n_components]])
        else:
            Y = np.concatenate(
                [embedding, 0.01 * tsvd.transform(X).astype(np.float32)])
    elif init == 'random':
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.concatenate([
            embedding, 0.0001 *
            np.random.normal(size=[X.shape[0], n_components]).astype(np.float32)
        ])
    else:     # user_supplied matrix
        init = init.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = np.concatenate([embedding, scaler.transform(Yinit) * 0.0001])

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1     # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(pair_XP.shape, verbose)

    for itr in range(num_iters):
        if itr < 100:
            w_neighbors = 2.0
        elif itr < 200:
            w_neighbors = 3.0
        else:
            w_neighbors = 1.0

        grad = pacmap_grad_fit(Y, pair_XP, w_neighbors)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    return Y, intermediate_states


class PaCMAP(Embedder):
    '''Pairwise Controlled Manifold Approximation.
    Maps high-dimensional dataset to a low-dimensional embedding.
    This class inherits the sklearn BaseEstimator, and we tried our best to
    follow the sklearn api. For details of this method, please refer to our publication:
    https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf
    Parameters
    ---------
    n_components: int, default=2
        Dimensions of the embedded space. We recommend to use 2 or 3.
    n_neighbors: int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.
    MN_ratio: float, default=0.5
        Ratio of mid near pairs to nearest neighbor pairs (e.g. n_neighbors=10, MN_ratio=0.5 --> 5 Mid near pairs)
        Mid near pairs are used for global structure preservation.
    FP_ratio: float, default=2.0
        Ratio of further pairs to nearest neighbor pairs (e.g. n_neighbors=10, FP_ratio=2 --> 20 Further pairs)
        Further pairs are used for both local and global structure preservation.
    pair_neighbors: numpy.ndarray, optional
        Nearest neighbor pairs constructed from a previous run or from outside functions.
    pair_MN: numpy.ndarray, optional
        Mid near pairs constructed from a previous run or from outside functions.
    pair_FP: numpy.ndarray, optional
        Further pairs constructed from a previous run or from outside functions.
    distance: string, default="euclidean"
        Distance metric used for high-dimensional space. Allowed metrics include euclidean, manhattan, angular, hamming.
    lr: float, default=1.0
        Learning rate of the Adam optimizer for embedding.
    num_iters: int, default=450
        Number of iterations for the optimization of embedding. 
        Due to the stage-based nature, we suggest this parameter to be greater than 250 for all three stages to be utilized.
    verbose: bool, default=False
        Whether to print additional information during initialization and fitting.
    apply_pca: bool, default=True
        Whether to apply PCA on the data before pair construction.
    intermediate: bool, default=False
        Whether to return intermediate state of the embedding during optimization.
        If True, returns a series of embedding during different stages of optimization.
    intermediate_snapshots: list[int], optional
        The index of step where an intermediate snapshot of the embedding is taken.
        If intermediate sets to True, the default value will be [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]
    random_state: int, optional
        Random state for the pacmap instance.
        Setting random state is useful for repeatability.
    save_tree: bool, default=False
        Whether to save the annoy index tree after finding the nearest neighbor pairs.
        Default to False for memory saving. Setting this option to True can make `transform()` method faster.
    '''

    def __init__(
        self,
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        pair_neighbors=None,
        pair_MN=None,
        pair_FP=None,
        distance="euclidean",
        lr=1.0,
        num_iters=450,
        verbose=False,
        apply_pca=True,
        intermediate=False,
        intermediate_snapshots=[
            0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450
        ],
        random_state=None,
        save_tree=False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.pair_neighbors = pair_neighbors
        self.pair_MN = pair_MN
        self.pair_FP = pair_FP
        self.distance = distance
        self.lr = lr
        self.num_iters = num_iters
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.intermediate = intermediate
        self.save_tree = save_tree
        self.intermediate_snapshots = intermediate_snapshots

        global _RANDOM_STATE
        if random_state is not None:
            assert (isinstance(random_state, int))
            self.random_state = random_state
            _RANDOM_STATE = random_state     # Set random state for numba functions
            warnings.warn(f'Warning: random state is set to {_RANDOM_STATE}')
        else:
            try:
                if _RANDOM_STATE is not None:
                    warnings.warn(f'Warning: random state is removed')
            except NameError:
                pass
            self.random_state = 0
            _RANDOM_STATE = None     # Reset random state

        if self.n_components < 2:
            raise ValueError(
                "The number of projection dimensions must be at least 2.")
        if self.lr <= 0:
            raise ValueError("The learning rate must be larger than 0.")
        if self.distance == "hamming" and apply_pca:
            warnings.warn(
                "apply_pca = True for Hamming distance. This option will be ignored."
            )
        if not self.apply_pca:
            warnings.warn(
                "Running ANNOY Indexing on high-dimensional data. Nearest-neighbor search may be slow!"
            )

    def decide_num_pairs(self, n):
        if self.n_neighbors is None:
            if n <= 10000:
                self.n_neighbors = 10
            else:
                self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        self.n_MN = int(round(self.n_neighbors * self.MN_ratio))
        self.n_FP = int(round(self.n_neighbors * self.FP_ratio))
        if self.n_neighbors < 1:
            raise ValueError(
                "The number of nearest neighbors can't be less than 1")
        if self.n_FP < 1:
            raise ValueError(
                "The number of further points can't be less than 1")

    def fit(self, X, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into a low-dimensional embedding, without returning the output.
        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.
        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.
        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        X = np.copy(X).astype(np.float32)
        # Preprocess the dataset
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        X, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
            X, self.distance, self.apply_pca, self.verbose, self.random_state,
            dim, self.n_components)
        self.tsvd_transformer = tsvd
        self.pca_solution = pca_solution
        # Deciding the number of pairs
        self.decide_num_pairs(n)
        print_verbose(
            "PaCMAP(n_neighbors={}, n_MN={}, n_FP={}, distance={}, "
            "lr={}, n_iters={}, apply_pca={}, opt_method='adam', "
            "verbose={}, intermediate={}, seed={})".format(
                self.n_neighbors, self.n_MN, self.n_FP, self.distance, self.lr,
                self.num_iters, self.apply_pca, self.verbose, self.intermediate,
                _RANDOM_STATE), self.verbose)
        # Sample pairs
        self.sample_pairs(X, self.save_tree)
        self.num_instances = X.shape[0]
        self.num_dimensions = X.shape[1]
        # Initialize and Optimize the embedding
        self.embedding_, self.intermediate_states, self.pair_neighbors, self.pair_MN, self.pair_FP = pacmap(
            X, self.n_components, self.pair_neighbors, self.pair_MN,
            self.pair_FP, self.lr, self.num_iters, init, self.verbose,
            self.intermediate, self.intermediate_snapshots, pca_solution,
            self.tsvd_transformer)
        if not save_pairs:
            self.del_pairs()
        return self

    def fit_transform(self, X, y=None, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into a low-dimensional embedding and return the embedding.
        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.
        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.
        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        self.fit(X, init, save_pairs)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def transform(self, X, basis=None, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into existing embedding space and return the embedding.
        Warning: In the current version of implementation, the `transform` method will treat the input as an 
        additional dataset, which means the same point could be mapped into a different place.
        Parameters
        ---------
        X: numpy.ndarray
            The new high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.
        basis: numpy.ndarray
            The original dataset that have already been applied during the `fit` or `fit_transform` process.
            If `save_tree == False`, then the basis is required to reconstruct the ANNOY tree instance.
            If `save_tree == True`, then it's unnecessary to provide the original dataset again.
        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset. 
            The PCA instance will be the same one that was applied to the original dataset during the `fit` or `fit_transform` process. 
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.
        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        # Preprocess the data
        X = np.copy(X).astype(np.float32)
        X = preprocess_X_new(X, self.distance, self.xmin, self.xmax, self.xmean,
                             self.tsvd_transformer, self.apply_pca,
                             self.verbose)
        if basis is not None and self.tree is None:
            basis = np.copy(basis).astype(np.float32)
            basis = preprocess_X_new(basis, self.distance, self.xmin, self.xmax,
                                     self.xmean, self.tsvd_transformer,
                                     self.apply_pca, self.verbose)
        # Sample pairs
        self.pair_XP = generate_extra_pair_basis(basis, X, self.n_neighbors,
                                                 self.tree, self.distance,
                                                 self.verbose)
        if not save_pairs:
            self.pair_XP = None

        # Initialize and Optimize the embedding
        Y, intermediate_states = pacmap_fit(
            X, self.embedding_, self.n_components, self.pair_XP, self.lr,
            self.num_iters, init, self.verbose, self.intermediate,
            self.intermediate_snapshots, self.pca_solution,
            self.tsvd_transformer)
        if self.intermediate:
            return intermediate_states
        else:
            return Y[self.embedding_.shape[0]:, :]

    def sample_pairs(self, X, save_tree):
        '''
        Sample PaCMAP pairs from the dataset.
        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected.
        save_tree: bool
            Whether to save the annoy index tree after finding the nearest neighbor pairs.
        '''
        # Creating pairs
        print_verbose("Finding pairs", self.verbose)
        if self.pair_neighbors is None:
            self.pair_neighbors, self.pair_MN, self.pair_FP, self.tree = generate_pair(
                X, self.n_neighbors, self.n_MN, self.n_FP, self.distance,
                self.verbose)
            print_verbose("Pairs sampled successfully.", self.verbose)
        elif self.pair_MN is None and self.pair_FP is None:
            print_verbose("Using user provided nearest neighbor pairs.",
                          self.verbose)
            assert self.pair_neighbors.shape == (
                X.shape[0] * self.n_neighbors, 2
            ), "The shape of the user provided nearest neighbor pairs is incorrect."
            self.pair_neighbors, self.pair_MN, self.pair_FP = generate_pair_no_neighbors(
                X, self.n_neighbors, self.n_MN, self.n_FP, self.pair_neighbors,
                self.distance, self.verbose)
            print_verbose("Pairs sampled successfully.", self.verbose)
        else:
            print_verbose("Using stored pairs.", self.verbose)

        if not save_tree:
            self.tree = None

        return self

    def del_pairs(self):
        '''
        Delete stored pairs.
        '''
        self.pair_neighbors = None
        self.pair_MN = None
        self.pair_FP = None
        return self
