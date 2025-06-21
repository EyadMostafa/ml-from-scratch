import numpy as np
import pandas as pd
from utils.optimizers import gradient_descent
from utils.helpers import validate_transform_input

class LinearSVC:
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self._w = None
        self._b = None
        self._X = None
        self._y = None

    _validate_transform_input = staticmethod(validate_transform_input)
    _gradient_descent = staticmethod(gradient_descent)

    def _hinge_loss_gradient(self, params):
        w, b = params

        margin = self._X @ w + b
        mask = margin <= 1

        dw = w - self.C * self._y[mask] @ self._X[mask]
        db = -self.C * np.sum(self._y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        self._X, self._y = self._validate_transform_input(X, y)
        self._y = np.where(self._y <= 0, -1, 1) 
        n = self._X.shape[1]

        self._w = np.zeros(n)
        self._b = 0.0

        self._w, self._b = self._gradient_descent(
            gradient_fn=self._hinge_loss_gradient,
            params=[self._w, self._b],
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
    def predict(self, X):
        X, _ = self._validate_transform_input(X)
        predictions = X @ self._w + self._b
        return np.where(predictions >= 0, 1, 0)
