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
    fit(X, y)
        Fit the LinearSVC model according to the given training data.
    
    predict(X)
        Predict class labels for samples in X.
    """
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000, tol=1e-4) -> None:
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w_ = None
        self.b_ = None

    __validate_transform_input = staticmethod(validate_transform_input)
    __gradient_descent = staticmethod(gradient_descent)

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

        dw = w - self.C * y[mask] @ X[mask]
        db = -self.C * np.sum(y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        X, y = self.__validate_transform_input(X, y)
        y = np.where(y <= 0, -1, 1) 
        n = X.shape[1]

        self.w_ = np.zeros(n)
        self.b_ = 0.0

        self.w_, self.b_ = self.__gradient_descent(
            gradient_fn=self._hinge_loss_gradient,
            params=np.array([self.w_, self.b_], dtype=object),
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            features=X,
            labels=y
        )
        
    def predict(self, X):
        if self.w_ is None or self.b_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self.__validate_transform_input(X)
        predictions = X @ self.w_ + self.b_
        return np.where(predictions >= 0, 1, 0)
