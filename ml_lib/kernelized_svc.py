import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from utils.helpers import validate_transform_input

class KernelizedSVC:
    """
    Support Vector Classifier with kernel support implemented from scratch.

    This class implements a binary classification Support Vector Machine (SVM)
    using different kernel functions and quadratic programming (via CVXOPT) to solve
    the dual optimization problem.

    Parameters
    ----------
    kernel : str, default='linear'
        The kernel type to be used in the algorithm. Must be one of:
        'linear', 'poly', or 'rbf'.
    
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely 
        proportional to C. Must be strictly positive.
    
    gamma : {'scale', float}, default='scale'
        Kernel coefficient for 'rbf' and 'poly'. If 'scale', it uses 
        1 / (n_features * X.var()).

    degree : int, default=3
        Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
    
    coef0 : float, default=0.0
        Independent term in polynomial kernel. Ignored by other kernels.

    Attributes
    ----------
    __X : ndarray
        Training feature matrix.

    __y : ndarray
        Training label vector (converted to -1 and 1).

    __kernel_matrix : ndarray
        Precomputed Gram matrix using the chosen kernel function.

    __alphas : ndarray
        Solution to the dual problem (Lagrange multipliers).

    __b : float
        Bias term of the decision function.

    __support_idx : ndarray
        Indices of support vectors used in bias computation.

    Methods
    -------
    fit(X, y)
        Fits the SVM model on training data.
    
    predict(X)
        Predicts class labels for input samples.

    get_params()
        Returns the current hyperparameters as a dictionary.

    set_params(**params)
        Sets the hyperparameters of the model.
    """
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0):
        self.__X = None
        self.__y = None
        self.__kernel = kernel
        self.__C = C
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
            'kernel': self.__kernel,
            'C': self.__C,
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
        """
        Compute the kernel (Gram) matrix between two datasets using the specified kernel.
    
        Parameters
        ----------
        X1 : ndarray of shape (n_samples_1, n_features)
            First input dataset.
    
        X2 : ndarray of shape (n_samples_2, n_features)
            Second input dataset.
    
        Returns
        -------
        K : ndarray of shape (n_samples_1, n_samples_2)
            Computed kernel matrix.
    
        Raises
        ------
        ValueError
        If an unsupported kernel type is specified.
        """
        if self.__kernel == 'linear':
            return X1 @ X2.T     
           
        elif self.__kernel == 'poly':
            return (self.__gamma * (X1 @ X2.T) + self.__coef0) ** self.__degree
        
        elif self.__kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.__gamma * sq_dists)

        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'poly', 'rbf', or 'sigmoid'.")


    def __compute_alpha(self):
        """
        Solve the dual optimization problem to compute Lagrange multipliers (alphas).
    
        This method uses the CVXOPT quadratic programming solver to solve the
        dual form of the SVM objective:
            maximize  L(α) = ∑α_i - 0.5 ∑∑α_i α_j y_i y_j K(x_i, x_j)
            subject to: 0 <= α_i <= C and ∑α_i y_i = 0
    
        Returns
        -------
        alphas : ndarray of shape (n_samples,)
            Lagrange multipliers for the support vectors.
        """
        m = self.__X.shape[0]
        self.__kernel_matrix = K = self.__compute_kernel_matrix(self.__X, self.__X)
        H = np.outer(self.__y, self.__y) * K
        f = -np.ones(m)
        A = np.vstack([np.eye(m), -np.eye(m)])
        b = np.hstack([self.__C * np.ones(m), np.zeros(m)])
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
        self.__support_idx = np.where((self.__alphas > 1e-5) & (self.__alphas < self.__C - 1e-5))[0]
        
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