import numpy as np
from .base_model import BaseModel
from utils.optimizers import gradient_descent

class LinearRegression(BaseModel):
    def __init__(self, fit_intercept=True, optimizer='normal', learning_rate=0.01, max_iter=1000, tol=1e-6):
        self._fit_intercept = fit_intercept
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol
        self._w = None
        self._b = None

    __gradient_descent = staticmethod(gradient_descent)

    def _mse_gradient(self, params, X, y):
        n_samples = X.shape[0]
        
        if self._fit_intercept:
            w, b = params
            y_pred = X @ w + b
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            return [dw, db] 
        else:
            (w,) = params
            y_pred = X @ w
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            return [dw]
        
    def _compute_weights(self, X, y):
        if self._fit_intercept:
            init_params = [self._w, self._b]
        else:
            init_params = [self._w]
    
        updated_params = self.__gradient_descent(
            gradient_fn=self._mse_gradient,
            params=init_params,
            features=X, 
            labels=y,
            learning_rate=self._learning_rate,
            max_iter=self._max_iter,
            tol=self._tol,
            fit_intercept=self._fit_intercept
        )
    
        if self._fit_intercept:
            self._w, self._b = updated_params
        else:
            self._w = updated_params[0]
            self._b = 0.0

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
        if self._optimizer == "normal":
            self._compute_normal(X, y)
        elif self._optimizer == "gd":
            self._w = np.zeros(X.shape[1])
            self._b = 0.0
            self._compute_weights(X, y)

    def predict(self, X):
        if self._w is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self._validate_transform_input(X)
        return X @ self._w + self._b


class RidgeRegression(LinearRegression):
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.__alpha = alpha

    def _mse_gradient(self, params, X, y):
        n_samples = X.shape[0]
        
        if self._fit_intercept:
            w, b = params
            y_pred = X @ w + b
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + self.__alpha * w
            db = (2 / n_samples) * np.sum(error)
            return [dw, db] 
        else:
            (w,) = params
            y_pred = X @ w
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + self.__alpha * w
            return [dw]

    def _compute_normal(self, X, y):
        if self._fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        penalty = self.__alpha * np.eye(X.shape[1])

        params = np.linalg.pinv(X.T @ X + penalty) @ X.T @ y

        if self._fit_intercept:
            self._b = params[0]
            self._w = params[1:]
        else:
            self._b = 0.0
            self._w = params 



class LassoRegression(LinearRegression):
    def __init__(self, alpha=0.1, fit_intercept=True, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__(fit_intercept=True, learning_rate=0.01, max_iter=1000, tol=1e-6)
        self.__alpha = alpha

    def _mse_gradient(self, params, X, y):
        n_samples = X.shape[0]
        
        if self._fit_intercept:
            w, b = params
            y_pred = X @ w + b
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + self.__alpha * np.sign(w)
            db = (2 / n_samples) * np.sum(error)
            return [dw, db] 
        else:
            (w,) = params
            y_pred = X @ w
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + self.__alpha * np.sign(w)
            return [dw]
        

class ElasticNetRegression(LinearRegression):
    def __init__(self, alpha=0.1, r=0.5, fit_intercept=True, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__(fit_intercept=True, learning_rate=0.01, max_iter=1000, tol=1e-6)
        self.__alpha = alpha
        self.__r = r

    def _mse_gradient(self, params, X, y):
        n_samples = X.shape[0]
        
        if self._fit_intercept:
            w, b = params
            l1_penalty = self.__r * self.__alpha * np.sign(w)
            l2_penalty = (1 - self.__r) * self.__alpha * w
            y_pred = X @ w + b
            error = y_pred - y


            dw = (2 / n_samples) * (X.T @ error) + l1_penalty + l2_penalty
            db = (2 / n_samples) * np.sum(error)
            return [dw, db] 
        else:
            (w,) = params
            l1_penalty = self.__r * self.__alpha * np.sign(w)
            l2_penalty = (1 - self.__r) * self.__alpha * w
            y_pred = X @ w
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error) + l1_penalty + l2_penalty
            return [dw]
