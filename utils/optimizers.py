import numpy as np

def gradient_descent(gradient_fn, params, features, labels, learning_rate=0.01, max_iter=1000, tol=1e-4, fit_intercept=True):    
    """
    Performs batch gradient descent optimization on a given objective.

    Parameters:
    -----------
    gradient_fn : callable
        A function that computes the gradients of the objective function. 
        Must accept (params, features, labels) as arguments and return a list/tuple of gradients.
    
    params : list or tuple of np.ndarray
        Initial parameters to be optimized (e.g., weights and bias).
    
    features : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    
    labels : np.ndarray
        The target labels of shape (n_samples,).
    
    learning_rate : float, default=0.01
        The step size to use during parameter updates.
    
    max_iter : int, default=1000
        The maximum number of iterations for gradient descent.
    
    tol : float, default=1e-4
        The tolerance for the stopping condition based on the L2 norm of the gradients.

    Returns:
    --------
    params : list of np.ndarray
        The optimized parameters after gradient descent.
    """
    for _ in range(max_iter):
        gradients = gradient_fn(params, features, labels)
        if fit_intercept:
            combined_grads = np.concatenate([gradients[0].ravel(), gradients[1].ravel()])
        else: combined_grads = [gradients[0].ravel()]
        if np.linalg.norm(combined_grads) < tol:
            break
        params = [p - g * learning_rate for p, g in zip(params, gradients)]
    return params