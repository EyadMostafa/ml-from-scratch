import numpy as np
from .helpers import validate_transform_input

class PCA:
    def __init__(self, n_components=None):
        self._n_components = n_components
        self._mean = None
        self._eigenvalues = None
        self._components = None

    _validate_tranform_input = staticmethod(validate_transform_input)

    def _compute_covariance_matrix(self, X):
        return (X.T @ X) / (X.shape[0] - 1)
    
    def _compute_eigens(self, cov_max):
        eigenvalues, eigenvectors = np.linalg.eig(cov_max)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors
    
    def fit(self, X):
        X, _ = self._validate_tranform_input(X)
        self._mean = X.mean(axis=0)
        X = X - self._mean
        cov_matrix = self._compute_covariance_matrix(X)
        eigenvalues, components = self._compute_eigens(cov_matrix)
        self._eigenvalues = eigenvalues[:self._n_components]
        self._components = components[:, :self._n_components]

    def transform(self, X):
        if self._components is None or self._eigenvalues is None:
            raise ValueError("PCA has not been fitted yet. Call 'fit' before 'tranform'.")
        X, _ = self._validate_tranform_input(X)
        X = X - self._mean
        return X @ self._components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
        

    

