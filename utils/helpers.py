import numpy as np
import pandas as pd

def validate_transform_input(X, y = None):
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


