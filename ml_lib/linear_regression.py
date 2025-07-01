import numpy as np
from .base_model import BaseModel
from utils.optimizers import gradient_descent

class LinearRegression(BaseModel):
    def __init__(self, fit_intercept=True, method='normal', learning_rate=0.01, n_iterations=1000, tol=1e-6):
        self._fit_intercept = fit_intercept
        self._method = method
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._tol = tol
        self._w = None
        self._b = None

    __gradient_descent = staticmethod(gradient_descent)

    def _compute_gradient(self, params, X, y):
        w, b = params
        n_samples = X.shape[0]
    
        y_pred = X @ w + b
        error = y_pred - y
    
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)
    
        return np.array([dw, db], dtype=object)

    def _compute_weights(self, X, y):
        self._w, self._b = self.__gradient_descent(
            gradient_fn=self._compute_gradient,
            params=np.array([self._w, self._b], dtype=object),
            features=X, 
            labels=y,
            learning_rate=self._learning_rate,
            max_iter=self._n_iterations,
            tol=self._tol)


    def _compute_normal(self, X, y):
        if self._fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        params = np.linalg.pinv(X.T @ X) @ X.T @ y

        if self._fit_intercept:
            self._b = params[0]
            self._w = params[1:]
        else:
            self._b = 0.0
            self._w = params 
        

    def fit(self, X, y):
        X, y = self._validate_transform_input(X, y)
        if self._method == "normal":
            self._compute_normal(X, y)
        elif self._method == "gradient":
            self._w = np.zeros(X.shape[1])
            self._b = 0.0
            self._compute_weights(X, y)

    def predict(self, X):
        if self._w is None or self._b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self._validate_transform_input(X)
        return X @ self._w + self._b
    


class RidgeRegression(LinearRegression):
    pass
