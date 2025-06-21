import numpy as np
import pandas as pd

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


