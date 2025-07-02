import numpy as np
from .base_model import BaseModel
from utils.optimizers import gradient_descent

## support quasi-newton lbfgs optimizer 

class LogisticRegression(BaseModel):
    def __init__(self, fit_intercept=True, 
                 learning_rate=0.01, max_iter=1000, 
                 tol=1e-6, C=1.0, optimizer="gd"):
        self._fit_intercept = fit_intercept
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol
        self._C = C
        self._optimizer = optimizer
        self._w = None
        self._b = None
        self.__mean = None
        self.__std = None

    def _compute_sigmoid(self, X, w, b):
        z = X @ w + b
        return 1 / (1 + np.exp(-z))
    
    def _log_loss_gradient(self, params, X, y):
        m = X.shape[0]
        if self._fit_intercept:
            w, b = params
        else:
            (w,) = params
            b = 0.0 
    
        y_pred = self._compute_sigmoid(X, w, b)
        error = y_pred - y
        penalty = (self._C / m) * w
        dw = (X.T @ error) / m + penalty
        db = np.sum(error) / m if self._fit_intercept else None
    
        return [dw, db] if self._fit_intercept else [dw]
    
    def _compute_normalization(self, X):
        self.__mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        self.__std = std
    
    def _normalize(self, X):
        return (X - self.__mean) / self.__std

    def fit(self, X, y):
        X, y = self._validate_transform_input(X, y)
        self._compute_normalization(X)
        X = self._normalize(X)

        if self._optimizer == "gd":
            self._w = np.zeros(X.shape[1])
            self._b = 0.0
            params = [self._w, self._b] if self._fit_intercept else [self._w]

            result = gradient_descent(
                gradient_fn=self._log_loss_gradient,
                params=params,
                features=X,
                labels=y,
                learning_rate=self._learning_rate,
                max_iter=self._max_iter,
                tol=self._tol,
                fit_intercept=self._fit_intercept
            )

            self._w, self._b = result
            self._b = 0.0 if not self._fit_intercept else self._b

    
    def predict_proba(self, X):
        if self._w is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self._validate_transform_input(X)
        X = self._normalize(X)

        z = X @ self._w + self._b

        return 1 / (1 + np.exp(-z)) 

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


    