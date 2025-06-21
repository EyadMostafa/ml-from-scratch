import numpy as np

def gradient_descent(gradient_fn, params, features, labels, learning_rate=0.01, max_iter=1000, tol=1e-4):
    for _ in range(max_iter):
        gradients = gradient_fn(params, features, labels)
        combined_grads = np.concatenate([gradients[0].ravel(), gradients[1].ravel()])
        if np.linalg.norm(combined_grads) < tol:
            break
        params = [p - g * learning_rate for p, g in zip(params, gradients)]
    return params