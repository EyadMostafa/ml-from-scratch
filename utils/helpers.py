import numpy as np
import pandas as pd
import matplotlib as plt

def validate_transform_input(X, y = None):
    """
    Validate and transform input features and labels into numpy arrays.

    Parameters
    ----------
    X : pd.DataFrame, pd.Series, or np.ndarray
        Feature data to be validated and converted.
    y : pd.DataFrame, pd.Series, np.ndarray, or None, default=None
        Target labels to be validated and converted.

    Returns
    -------
    tuple
        Tuple containing:
        - X: numpy.ndarray of shape (n_samples, n_features)
        - y: numpy.ndarray of shape (n_samples,) or None if y was None

    Raises
    ------
    ValueError
        If inputs are not pandas DataFrame, Series, or numpy ndarray.
        If y is not 1-dimensional.
        If number of samples in X and y do not match.

    Notes
    -----
    - If y has shape (n_samples, 1), it is flattened to (n_samples,).
    - If y is None, returns (X, None).
    """
    if(isinstance(X, (pd.DataFrame, pd.Series))):
        X = X.values
    elif not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a pandas DataFrame, Series, or a numpy array.")
    
    if y is None:
        return X, None
    
    if(isinstance(y, (pd.DataFrame, pd.Series))):
        y = y.values
    elif not isinstance(y, np.ndarray):
        raise ValueError("Input y must be a pandas DataFrame, Series, or a numpy array.")
    
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError("Input y must be a 1-dimensional array.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y must match.")
    
    return X, y


def plot_decision_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].min() + 0.5 
    y_min, y_max = X[:, 1].max() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max))

    Z = clf.predict(np.c_(xx.ravel(), yy.ravel()))
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    colors = ["blue", "red"]

    for val in np.unique(y):
        ax.scatter(X[y==val, 0], X[y==val, 1], c=colors[val], label="Class {val}", edgecolor="-k") 

    ax.set_title(title)
    ax.legend()

def plot_decision_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    colors = ['blue', 'red']

    for val in np.unique(y):
        ax.scatter(X[y == val, 0], X[y == val, 1], c=colors[val], label=f'Class {val}', edgecolor='k')
    
    ax.set_title(title)
    ax.legend()

