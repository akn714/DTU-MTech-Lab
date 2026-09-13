import numpy as np


class MyPCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X):
        # Assumes X is already centered and scaled
        # np.cov expects variables as rows and observations as columns
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors for symmetric covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns eigenvalues in ascending order; reverse for descending
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Store top n_components eigenvectors (columns)
        self.components_ = sorted_eigenvectors[:, : self.n_components]
        return self

    def transform(self, X):
        # Project data onto principal component subspace
        # Take real part to protect against minute numerical artifacts
        return np.real(np.dot(X, self.components_))