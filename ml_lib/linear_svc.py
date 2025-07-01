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
    def __init__(self, C=1.0, learning_rate=0.01, max_iter=1000, tol=1e-4) -> None:
        self.__C = C
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__tol = tol
        self.__w = None
        self.__b = None

    __gradient_descent = staticmethod(gradient_descent)

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

        dw = w - self.__C * y[mask] @ X[mask]
        db = -self.__C * np.sum(y[mask])

        return np.array([dw, db], dtype=object)

    def fit(self, X, y):
        """
        Train the linear SVM using gradient descent.

        Parameters:
            X (ndarray): Training feature matrix.
            y (ndarray): Binary target labels (0 or 1).
        """
        if len(np.unique(y)) != 2:
            raise ValueError("Kernelized SVC only supports binary classification.")
        X, y = self._validate_transform_input(X, y)
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
        """
        Predict binary class labels for input samples.

        Parameters:
            X (ndarray): Input feature matrix.

        Returns:
            preds (ndarray): Predicted class labels (0 or 1).
        """
        X, _ = self._validate_transform_input(X)
        if self.__w is None or self.__b is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        predictions = X @ self.__w + self.__b
        return np.where(predictions >= 0, 1, 0)
