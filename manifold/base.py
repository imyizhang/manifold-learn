from typing import Optional

import numpy
import torch


class Estimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> "Estimator":
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


class Wrapper(object):
    def __init__(self, estimator: Estimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        X: numpy.ndarray,
        y: Optional[numpy.ndarray] = None,
    ) -> "Wrapper":
        """Fit the model with X."""
        X = torch.from_numpy(X)
        if y is not None:
            y = torch.from_numpy(y)
        return self.__class__(self.estimator.fit(X, y=y))

    def fit_transform(
        self,
        X: numpy.ndarray,
        y: Optional[numpy.ndarray] = None,
        **fit_params,
    ) -> numpy.ndarray:
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
        X = torch.from_numpy(X)
        if y is not None:
            y = torch.from_numpy(y)
        return (
            self.estimator.fit_transform(X, y=y, **fit_params)
            .detach()
            .cpu()
            .numpy()
        )

    def transform(self, X: numpy.ndarray) -> numpy.ndarray:
        """Apply dimensionality reduction to X."""
        X = torch.from_numpy(X)
        return self.estimator.transform(X).detach().cpu().numpy()
