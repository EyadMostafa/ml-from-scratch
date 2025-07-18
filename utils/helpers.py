import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (make_scorer,
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score,
                             confusion_matrix,
                             silhouette_score,
                             silhouette_samples,
                             adjusted_rand_score)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


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


def evaluate_metrics(y_true, y_pred, X=None, title="Model Evaluation", average="binary", supervised=True):
    if supervised:
        results = pd.DataFrame([{
            "Model": title,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        }])
    else:
        results = pd.DataFrame([{
            "Model": title,
            "Silouhette": silhouette_score(X, y_pred),
            "Adjusted-Rand": adjusted_rand_score(y_true, y_pred),
        }])  

    return results


def plot_confusion_matrix(y_true, y_pred, ax, labels=None, normalize=False, 
                          title="Confusion Matrix", cmap="Blues", xvisible=True, yvisible=True, cbar=True):

    labels = labels if labels else sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fmt = ".2f" if normalize else "d"
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax, cbar=cbar)

    if not xvisible:
        ax.get_xaxis().set_visible(False)
    if not yvisible:
        ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def train_time(clf, X, y=None, title="Model", verbose=True, supervised=True):
    start = time.time()
    if supervised:
        clf.fit(X, y)
    else: clf.fit(X)
    duration = time.time() - start

    if verbose:
        print(f"{title} trained in {duration:.4f} seconds.")

    return np.round(duration, 4)

        
def plot_decision_boundary(
    clf, X, y=None, ax=None, title="Decision Boundary",
    cmap=plt.cm.coolwarm, colors=['blue', 'red'],
    xvisible=True, yvisible=True, s=30,
    label_points=True,
    legend=True,
    alpha=0.8
):
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap)

    if y is not None:
        for idx, val in enumerate(np.unique(y)):
            ax.scatter(X[y == val, 0], X[y == val, 1], c=colors[idx], 
                       label=f'Class {val}' if label_points else None, edgecolor='k', s=s)
        if legend:
            ax.legend()
    else:
        ax.scatter(X[:, 0], X[:, 1], c='black', edgecolor='k', s=s)

    if not xvisible:
        ax.get_xaxis().set_visible(False)
    if not yvisible:
        ax.get_yaxis().set_visible(False)

    ax.set_title(title)



def cross_validate(model_class, X, y, cv=5, scoring_metrics=None, seed=42, verbose=True, **model_params):
    """
    Custom k-fold cross-validation using NumPy (no sklearn splitters).

    Parameters
    ----------
    model_class : class
        Your custom model class.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int
        Number of folds.
    scoring_metrics : list of str
        Metrics to compute: ['accuracy', 'precision', 'recall', 'f1']
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print per-fold results.
    model_params : kwargs
        Parameters passed to your model constructor.

    Returns
    -------
    pd.DataFrame
        DataFrame of averaged scores across folds.
    """
    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    metric_funcs = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

    # Check for valid metrics
    for metric in scoring_metrics:
        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric: {metric}")

    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)

    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1

    current = 0
    fold_scores = {metric: [] for metric in scoring_metrics}

    for fold in range(cv):
        start, stop = current, current + fold_sizes[fold]
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if verbose:
            print(f"\nFold {fold + 1}:")

        for metric in scoring_metrics:
            score = metric_funcs[metric](y_test, y_pred)
            fold_scores[metric].append(score)
            if verbose:
                print(f"{metric.capitalize()}: {score:.4f}")

        current = stop

    # Compute average scores
    avg_scores = {metric: np.mean(fold_scores[metric]) for metric in scoring_metrics}
    return pd.DataFrame([avg_scores])



def plot_silhouette_chart(X, y, ax, n_clusters, title="Model Silhouette Chart"):

    silhouette = silhouette_score(X, y)
    instance_silhouette_score = silhouette_samples(X, y)

    silhouette_avgs = {}
    for i in range(n_clusters):
        silhouette_avgs[i] = np.mean(instance_silhouette_score[y == i])

    silhouette_avgs = sorted(silhouette_avgs.items(), 
                             key=lambda item: item[1],
                             reverse=True)

    y_lower = 10

    for i, _ in silhouette_avgs:

        cluster_silhouette = instance_silhouette_score[y == i]
        cluster_silhouette.sort()

        size = cluster_silhouette.shape[0]
        y_upper = y_lower + size

        color = cm.nipy_spectral(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size, str(i))

        y_lower = y_upper + 10

    ax.axvline(x=silhouette, color='red', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])


def visualize_clustering(X, y, 
                         centroids=None, 
                         ax=None, title="Model Clusters", 
                         xvisible=True, yvisible=True,
                         label_points=True,
                         legend=True):

    if ax is None:
        fig, ax = plt.subplots()

    unique_labels = np.unique(y)
    
    colors = [cm.nipy_spectral(i / len(unique_labels)) for i in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        cluster_mask = (y == label)
        ax.scatter(*X[cluster_mask].T,
                   color=colors[i],
                   label=f"Cluster {label}" if label_points else None,
                   alpha=0.8)
        
    if centroids is not None and len(centroids) > 0:
            ax.scatter(*centroids.T, 
                       c="red",
                       marker="x")
        
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if legend:
        ax.legend(title="Clusters")
    if not xvisible:
        ax.get_xaxis().set_visible(False)
    if not yvisible:
        ax.get_yaxis().set_visible(False)