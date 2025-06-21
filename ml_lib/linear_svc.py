import numpy as np
import pandas as pd
from utils.optimizers import gradient_descent
from utils.helpers import validate_transform_input

class LinearSVC:
    """
    Linear Support Vector Classifier using hinge loss and gradient descent optimization.

    Parameters:
    -----------
    C : float
        Regularization parameter.
    learning_rate : float
        Learning rate for gradient descent.
    max_iter : int
        Maximum number of iterations for gradient descent.
    tol : float
        Tolerance for stopping criteria based on gradient norm.
    """
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000, tol=1e-4) -> None:
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self._w = None
        self._b = None

    __validate_transform_input = staticmethod(validate_transform_input)
    __gradient_descent = staticmethod(gradient_descent)

    def _hinge_loss_gradient(self, params, X, y):
        w, b = params

        margin = X @ w + b
        mask = margin <= 1

        dw = w - self.C * y[mask] @ X[mask]
        db = -self.C * np.sum(y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        X, y = self.__validate_transform_input(X, y)
        y = np.where(y <= 0, -1, 1) 
        n = X.shape[1]

        self._w = np.zeros(n)
        self._b = 0.0

        self._w, self._b = self.__gradient_descent(
            gradient_fn=self._hinge_loss_gradient,
            params=[self._w, self._b],
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            features=X,
            labels=y
        )
        
    def predict(self, X):
        if self._w is None or self._b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self.__validate_transform_input(X)
        predictions = X @ self._w + self._b
        return np.where(predictions >= 0, 1, 0)
