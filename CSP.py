import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator


class CSP(TransformerMixin, BaseEstimator):
    """
    Common Spatial Patterns (CSP) is a spatial filtering method that finds
    projection vectors that maximize the variance
    of one class while minimizing the variance of the other class.

    Attributes
    ----------
    n_components : int
        Number of components to keep
    filters_ : array, shape (n_components, n_channels)
        CSP filters

    Methods
    -------
    fit(X, y)
        Fits CSP filters to the data
    transform(X)
        Transforms the data to CSP space
    fit_transform(X, y)
        Fits CSP filters to the data and transforms it to CSP space
    _compute_covariance_matrices(X, y)
        Computes covariance matrices for each class

    Raises
    ------
    ValueError
        If n_components is not an int
    """

    def __init__(self, n_components: int = 6):
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int")

        self.n_components = n_components

    def _compute_covariance_matrices(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes covariance matrices for each class

        Parameters
        ----------
        X : np.ndarray
            Data
        y : np.ndarray
            Labels

        Returns
        -------
        np.ndarray
            Covariance matrices for each class
        """
        covs = []
        for cur_class in self._classes:
            x_class = X[y == cur_class]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(x_class.shape[0], -1)
            cov_mat = np.cov(x_class)
            covs.append(cov_mat)
        return np.array(covs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        """
        Fits CSP filters to the data

        Parameters
        ----------
        X : np.ndarray
            Data
        y : np.ndarray
            Labels

        Returns
        -------
        CSP
            CSP object
        """
        self._classes = np.unique(y)
        covs = self._compute_covariance_matrices(X, y)

        eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        self.filters_ = eigen_vectors[:, ix].T[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data to CSP space

        Parameters
        ----------
        X : np.ndarray
            Data

        Returns
        -------
        np.ndarray
            Data in CSP space
        """
        X_transformed = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        X_transformed = (X_transformed**2).mean(axis=2)
        X_transformed -= X_transformed.mean()
        X_transformed /= X_transformed.std()
        return X_transformed

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fits CSP filters to the data and transforms it to CSP space

        Parameters
        ----------
        X : np.ndarray
            Data
        y : np.ndarray
            Labels

        Returns
        -------
        np.ndarray
            Data in CSP space
        """
        return self.fit(X, y).transform(X)
