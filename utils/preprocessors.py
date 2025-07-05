import numpy as np
from .helpers import validate_transform_input

class PCA:
    def __init__(self, n_components=None):
        self.__n_components = n_components
        self.__mean = None
        self.__eigenvalues = None
        self.__components = None

    _validate_tranform_input = staticmethod(validate_transform_input)

    def __compute_covariance_matrix(self, X):
        return (X.T @ X) / (X.shape[0] - 1)
    
    def __compute_eigens(self, cov_max):
        eigenvalues, eigenvectors = np.linalg.eig(cov_max)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors
    
    def fit(self, X):
        X, _ = self._validate_tranform_input(X)
        self.__mean = X.mean(axis=0)
        X = X - self.__mean
        cov_matrix = self.__compute_covariance_matrix(X)
        eigenvalues, components = self.__compute_eigens(cov_matrix)
        self.__eigenvalues = eigenvalues[:self.__n_components]
        self.__components = components[:, :self.__n_components]

    def transform(self, X):
        if self.__components is None or self.__eigenvalues is None:
            raise ValueError("PCA has not been fitted yet. Call 'fit' before 'tranform'.")
        X, _ = self._validate_tranform_input(X)
        X = X - self.__mean
        return X @ self.__components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
        

    

