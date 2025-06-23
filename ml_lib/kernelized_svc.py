import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from utils.helpers import validate_transform_input

class KernelizedSVC:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0):
        self.__X = None
        self.__y = None
        self.kernel = kernel
        self.C = C
        self.__gamma = gamma
        self.__degree = degree
        self.__coef0 = coef0
        self.__kernel_matrix = None
        self.__alphas = None
        self.__b = None
        self.__support_idx = None

    __validate_transform_input = staticmethod(validate_transform_input)

    def get_params(self):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.__gamma,
            'degree': self.__degree,
            'coef0': self.__coef0
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, f'_{self.__class__.__name__}__{key}'):
                setattr(self, f'_{self.__class__.__name__}__{key}', value)
            else:
                raise ValueError(f"Parameter '{key}' is not valid for {self.__class__.__name__}.")
        return self

    def __compute_kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T        
        elif self.kernel == 'poly':
            return (self.__gamma * (X1 @ X2.T) + self.__coef0) ** self.__degree
        
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.__gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'poly', or 'rbf'.")


    def __compute_alpha(self):
        m = self.__X.shape[0]
        self.__kernel_matrix = K = self.__compute_kernel_matrix(self.__X, self.__X)
        H = np.outer(self.__y, self.__y) * K
        f = -np.ones(m)
        A = np.vstack([np.eye(m), -np.eye(m)])
        b = np.hstack([self.C * np.ones(m), np.zeros(m)])
        A_eq = self.__y.reshape(1, -1)
        b_eq = np.array([0.0])

        H = matrix(H)
        f = matrix(f)
        A = matrix(A)
        b = matrix(b)
        A_eq = matrix(A_eq.astype('double'))
        b_eq = matrix(b_eq.astype('double'))

        solvers.options['show_progress'] = False
        solution = solvers.qp(H, f, A, b, A_eq, b_eq)
        alphas = np.array(solution['x']).flatten()
        return alphas
    
    def fit(self, X, y):
        self.__X, self.__y = self.__validate_transform_input(X, y)
        if len(np.unique(y)) != 2:
            raise ValueError("Kernelized SVC only supports binary classification.")
        
        if self.__gamma == 'scale':
            self.__gamma = 1.0 / (self.__X.shape[1] * self.__X.var())

        self.__y = np.where(self.__y <= 0, -1, 1)
        self.__alphas = self.__compute_alpha()
        self.__support_idx = np.where((self.__alphas > 1e-5) & (self.__alphas < self.C - 1e-5))[0]
        
        b_vals = []
        for i in self.__support_idx:
            b_i = self.__y[i] - np.sum(self.__alphas * self.__y * self.__kernel_matrix[i])
            b_vals.append(b_i)
        self.__b = np.mean(b_vals)


    def predict(self, X):
        X, _ = self.__validate_transform_input(X)
        if self.__alphas is None or self.__b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        kernel_matrix_test = self.__compute_kernel_matrix(self.__X, X)
        predictions = kernel_matrix_test.T @ (self.__alphas * self.__y) + self.__b
        return np.where(predictions >= 0, 1, 0)