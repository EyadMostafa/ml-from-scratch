import numpy as np
import pandas as pd
from utils.optimizers import gradient_descent
from utils.helpers import validate_transform_input

class LinearSVC:
    """
    Linear Support Vector Classifier implemented with batch gradient descent and hinge loss.

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C.
    
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    
    max_iter : int, default=1000
        Maximum number of iterations for the gradient descent optimization.
    
    tol : float, default=1e-4
        Tolerance for the stopping criterion. Optimization stops when the norm of the gradient is below this value.

    Attributes:
    -----------
    _w : np.ndarray
        Learned weight vector after fitting.
    
    _b : float
        Learned bias term after fitting.

    Methods:
    --------
    _hinge_loss_gradient(params, X, y)
        Compute the gradient of the hinge loss plus L2 regularization with respect to weights and bias.

    fit(X, y)
        Fit the LinearSVC model according to the given training data.
    
    predict(X)
        Predict class labels for samples in X.
    """
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000, tol=1e-4) -> None:
        self.__C = C
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__tol = tol
        self.__w = None
        self.__b = None

    __validate_transform_input = staticmethod(validate_transform_input)
    __gradient_descent = staticmethod(gradient_descent)

    def get_params(self):
        return {
            'C' : self.__C,
            'learning_rate': self.__learning_rate,
            'max_iter': self.__max_iter,
            'tol': self.__tol,
            'w': self.__w, 
            'b': self.__b
        }
    
    def set_params(self, **params):
        for key, value in params.items():
           if(hasattr(self, f'_{self.__class__.__name__}__{key}')):
               setattr(self, f'_{self.__class__.__name__}__{key}', value)
           else:
               raise ValueError(f"Parameter '{key}' is not valid for {self.__class__.__name__}.")
        return self

    def _hinge_loss_gradient(self, params, X, y):
        """
        Compute the gradient of the hinge loss plus L2 regularization with respect to weights and bias.
    
        Parameters:
        -----------
        params : list or tuple of np.ndarray
            Current parameters [weights, bias].
        
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        
        y : np.ndarray
            Target labels encoded as {-1, 1} (n_samples,).
    
        Returns:
        --------
        gradients : np.ndarray
            Gradients for weights and bias as a numpy array of objects [dw, db].
       """
        w, b = params

        margin = X @ w + b
        mask = margin <= 1

        dw = w - self.__C * y[mask] @ X[mask]
        db = -self.__C * np.sum(y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        if len(np.unique(y)) != 2:
            raise ValueError("Kernelized SVC only supports binary classification.")
        X, y = self.__validate_transform_input(X, y)
        y = np.where(y <= 0, -1, 1) 
        n = X.shape[1]

        self.__w = np.zeros(n)
        self.__b = 0.0

        self.__w, self.__b = self.__gradient_descent(
            gradient_fn=self._hinge_loss_gradient,
            params=np.array([self.__w, self.__b], dtype=object),
            learning_rate=self.__learning_rate,
            max_iter=self.__max_iter,
            tol=self.__tol,
            features=X,
            labels=y
        )
        
    def predict(self, X):
        X, _ = self.__validate_transform_input(X)
        if self.__w is None or self.__b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        predictions = X @ self.__w + self.__b
        return np.where(predictions >= 0, 1, 0)
