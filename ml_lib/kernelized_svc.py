import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from utils.helpers import validate_transform_input

class KernelizedSVC:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.__gamma = 0.01
        self.__degree = 3
        self.__coef0 = 1.0
        self.__alpha = None

    def __compute_kernel_matrix(self, X):
        if self.kernel == 'linear':
            return X @ X.T
        
        elif self.kernel == 'poly':
            return (self.__gamma * (X @ X.T) + self.__coef0) ** self.__degree
        
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * (X @ X.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.__gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'poly', or 'rbf'.")

