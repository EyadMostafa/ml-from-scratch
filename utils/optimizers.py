import numpy as np

def gradient_descent(gradient_fn, params, features, labels, learning_rate=0.01, max_iter=1000, tol=1e-4, fit_intercept=True):    
    w, b = params if fit_intercept else (params[0], 0.0)

    for _ in range(max_iter):
        gradients = gradient_fn([w, b] if fit_intercept else [w], features, labels)

        dw = gradients[0]
        db = gradients[1] if fit_intercept else 0.0

        # Update parameters
        w -= learning_rate * dw
        if fit_intercept:
            b -= learning_rate * db

        # Check convergence
        grad_norm = np.linalg.norm(np.concatenate([dw.ravel(), np.array([db])]) if fit_intercept else dw.ravel())
        if grad_norm < tol:
            break

    return (w, b) if fit_intercept else (w, None)