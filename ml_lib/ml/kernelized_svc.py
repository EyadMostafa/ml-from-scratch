import numpy as np
from cvxopt import matrix, solvers
from .base_model import BaseModel

class KernelizedSVC(BaseModel):
    """
    Support Vector Classifier with kernel support.

    This binary classifier solves the dual form of the SVM problem using quadratic programming 
    and supports multiple kernel types.

    Parameters:
        kernel (str): Kernel type to use. Options: 'linear', 'poly', 'rbf'.
        C (float): Regularization parameter. Must be > 0.
        gamma (float or 'scale'): Kernel coefficient for 'rbf' and 'poly'.
        degree (int): Degree of the polynomial kernel (only used if kernel='poly').
        coef0 (float): Independent term in polynomial kernel (only used if kernel='poly').

    Attributes:
        _X (ndarray): Training features.
        _y (ndarray): Binary training labels (converted to -1 and 1).
        _alphas (ndarray): Lagrange multipliers.
        _b (float): Bias term.
        _support_indices_vis = (ndarray): Indices used for visualization since they match sklearn's indices.
        __support_indices (ndarray): Indices of used for actual computation since the lead to more accurate results.
        _kernel_matrix (ndarray): Precomputed Gram matrix.

    Methods:
        fit(X, y): Train the SVM on input data.
        predict(X): Predict class labels for input samples.
        get_params(): Return model hyperparameters.
        set_params(**params): Update model hyperparameters.
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0):
        self._X = None
        self._y = None
        self._kernel = kernel
        self._C = C
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0
        self._kernel_matrix = None
        self._alphas = None
        self._b = None
        self._support_indices_vis = None
        self.__support_indices = None

    def __compute_kernel_matrix(self, X1, X2):
        """
        Compute the kernel (Gram) matrix using the selected kernel.

        Parameters:
            X1 (ndarray): Input data of shape (n_samples_1, n_features).
            X2 (ndarray): Input data of shape (n_samples_2, n_features).

        Returns:
            K (ndarray): Kernel matrix of shape (n_samples_1, n_samples_2).
        """
        if self._kernel == 'linear':
            return X1 @ X2.T     
           
        elif self._kernel == 'poly':
            return (self._gamma * (X1 @ X2.T) + self._coef0) ** self._degree
        
        elif self._kernel == 'rbf':
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self._gamma * sq_dists)

        else:
            raise ValueError("Unsupported kernel type. Use 'linear', 'poly', 'rbf'.")


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
        m = self._X.shape[0]
        self._kernel_matrix = K = self.__compute_kernel_matrix(self._X, self._X)
        H = np.outer(self._y, self._y) * K
        f = -np.ones(m)
        A = np.vstack([np.eye(m), -np.eye(m)])
        b = np.hstack([self._C * np.ones(m), np.zeros(m)])
        A_eq = self._y.reshape(1, -1)
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
        """
        Train the kernel SVM on input data.

        Parameters:
            X (ndarray): Training feature matrix.
            y (ndarray): Training labels (binary).
        """
        self._X, self._y = self._validate_transform_input(X, y)
        if len(np.unique(y)) != 2:
            raise ValueError("Kernelized SVC only supports binary classification.")
        
        if self._gamma == 'scale':
            self._gamma = 1.0 / (self._X.shape[1] * self._X.var())

        self._y = np.where(self._y <= 0, -1, 1)
        self._alphas = self.__compute_alpha()
        self.__support_indices = np.where((self._alphas > 1e-5) & (self._alphas < self._C - 1e-5))[0]
        self._support_indices_vis = np.where(self._alphas > 1e-6)[0]
        
        b_vals = []
        for i in self.__support_indices:
            b_i = self._y[i] - np.sum(self._alphas * self._y * self._kernel_matrix[i])
            b_vals.append(b_i)
        self._b = np.mean(b_vals)


    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
            X (ndarray): Input samples of shape (n_samples, n_features).

        Returns:
            preds (ndarray): Predicted class labels (0 or 1).
        """
        X, _ = self._validate_transform_input(X)
        if self._alphas is None or self._b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        kernel_matrix = self.__compute_kernel_matrix(self._X, X)
        predictions = kernel_matrix.T @ (self._alphas * self._y) + self._b
        return np.where(predictions >= 0, 1, 0)