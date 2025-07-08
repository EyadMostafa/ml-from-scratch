import numpy as np
from utils.optimizers import gradient_descent
from .base_model import BaseModel

class LinearSVC(BaseModel):
    """
    Linear Support Vector Classifier.

    This model minimizes hinge loss with L2 regularization using batch gradient descent.
    It supports binary classification and learns a linear decision boundary.

    Parameters:
        C (float): Regularization strength (inverse). Higher values reduce regularization.
        learning_rate (float): Step size for gradient descent updates.
        max_iter (int): Maximum number of gradient descent iterations.
        tol (float): Tolerance for stopping criteria based on gradient norm.

    Attributes:
        _w (ndarray): Learned weight vector.
        _b (float): Learned bias term.

    Methods:
        fit(X, y): Train the model using hinge loss on binary-labeled data.
        predict(X): Predict class labels for input samples.
    """
    def __init__(self, C=1.0, learning_rate=1e-6, max_iter=1000, tol=1e-4) -> None:
        self._C = C
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._tol = tol
        self._w = None
        self._b = None

    def _hinge_loss_gradient(self, params, X, y):
        """
        Compute gradients of hinge loss with L2 regularization.

        Parameters:
            params (list or tuple): Current model parameters [weights, bias].
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Binary labels in {-1, 1} of shape (n_samples,).

        Returns:
            gradients (ndarray): Gradients [dw, db] as an array of objects.
        """
        w, b = params

        margin = X @ w + b
        mask = margin <= 1

        dw = w - self._C * y[mask] @ X[mask]
        db = -self._C * np.sum(y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        """
        Train the linear SVM using gradient descent.

        Parameters:
            X (ndarray): Training feature matrix.
            y (ndarray): Binary target labels (0 or 1).
        """
        if len(np.unique(y)) != 2:
            raise ValueError("Linear SVC only supports binary classification.")
        X, y = self._validate_transform_input(X, y)
        y = np.where(y <= 0, -1, 1) 
        n = X.shape[1]

        self._w = np.zeros(n)
        self._b = 0.0

        self._w, self._b = gradient_descent(
            gradient_fn=self._hinge_loss_gradient,
            params=np.array([self._w, self._b], dtype=object),
            learning_rate=self._learning_rate,
            max_iter=self._max_iter,
            tol=self._tol,
            features=X,
            labels=y
        )

        return self
        
    def predict(self, X):
        """
        Predict binary class labels for input samples.

        Parameters:
            X (ndarray): Input feature matrix.

        Returns:
            preds (ndarray): Predicted class labels (0 or 1).
        """
        X, _ = self._validate_transform_input(X)
        if self._w is None or self._b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        predictions = X @ self._w + self._b
        return np.where(predictions >= 0, 1, 0)
