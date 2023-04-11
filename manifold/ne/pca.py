import torch
from torch.linalg import svd

from manifold.base import Estimator

# create a random dataset with 100 samples and 20 features
X = torch.randn(100, 20)

# center the data
X_mean = torch.mean(X, dim=0, keepdim=True)
X_centered = X - X_mean

# perform SVD
U, S, V = svd(X_centered)

# choose the number of principal components to keep
k = 5

# compute the reduced dimensionality representation of the data
X_reduced = torch.mm(X_centered, V[:, :k])

# print the variance explained by each principal component
explained_variance = torch.pow(S[:k], 2) / (X_centered.shape[0] - 1)
explained_variance_ratio = explained_variance / torch.sum(explained_variance)


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def svd():
    return


def pca(X: torch.Tensor, n_components: int) -> torch.Tensor:
    return PCA(n_components).fit_transform(X)


class PCA(Estimator):

    def __init__(
        self,
        n_components: int = 2,
    ):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_     # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
