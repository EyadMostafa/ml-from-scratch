import numpy as np
from .base_model import BaseModel
from utils.helpers import validate_transform_input


class KMeans(BaseModel):
    def __init__(self, n_clusters=8, max_iter=int(1e9), tol=0):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._inertia = None
        self._centroids = None
        self._labels  = None
        self._centroids_over_iters = []
        self._labels_over_iters = []
    
    def __compute_squared_euclidean_dist(self, X, centroid):
        return np.sum((X - centroid)**2, axis=1)
    
    def __compute_inertia(self, X, centroids, labels):
        inertia = 0
        for i in range(self._n_clusters ):
            inertia += np.sum(self.__compute_squared_euclidean_dist(X[labels == i], centroids[i]))

        return inertia

    def __compute_initial_centroids(self, X):
        m = X.shape[0]
        centroids = []
        D = np.full(m, np.inf)
        init_centroid_idx = np.random.choice(m)
        init_centroid = X[init_centroid_idx]
        centroids.append(init_centroid)

        for _ in range(self._n_clusters  - 1):
            D = np.minimum(D, self.__compute_squared_euclidean_dist(X, centroids[-1]))
            probs = D / np.sum(D)
            new_centroid_idx = np.random.choice(m, p=probs)
            new_centroid = X[new_centroid_idx]
            centroids.append(new_centroid)

        return np.array(centroids)


    def __compute_labels(self, X, centroids):
        distances = np.vstack([self.__compute_squared_euclidean_dist(X, centroids[i]) 
                     for i in range(self._n_clusters)])
        return np.argmin(distances, axis=0)

    def __compute_centroids(self, X):
        for _ in range(int(self._max_iter)):
            centroids = []
            for i in range(self._n_clusters):
                cluster_points = X[self._labels  == i]
                if len(cluster_points) == 0:
                    centroids.append(X[np.random.choice(X.shape[0])])
                else:
                    centroids.append(np.mean(cluster_points, axis=0))
           
            new_labels = self.__compute_labels(X, centroids)
            new_inertia = self.__compute_inertia(X, centroids, new_labels)
            if self._inertia > new_inertia and abs(self._inertia - new_inertia) > self._tol:
                self._inertia = new_inertia
                self._centroids = np.array(centroids)
                self._labels  = new_labels
                self._centroids_over_iters.append(centroids)
                self._labels_over_iters.append(new_labels)
            else: 
                break

    def fit(self, X):
        X, _ = self._validate_transform_input(X)
        self._centroids = self.__compute_initial_centroids(X)
        self._centroids_over_iters.append(self._centroids)
        self._labels  = self.__compute_labels(X, self._centroids)
        self._labels_over_iters.append(self._labels)
        self._inertia = self.__compute_inertia(X, self._centroids, self._labels )
        self.__compute_centroids(X)

        self._centroids_over_iters = np.array(self._centroids_over_iters)
        self._labels_over_iters = np.array(self._labels_over_iters)

        return self

    def transform(self, X):
        X, _ = self._validate_transform_input(X)
        distances = np.zeros((X.shape[0], self._n_clusters))
        for i in range(self._n_clusters):
            distances[:,i] = self.__compute_squared_euclidean_dist(X, self._centroids[i])
        return distances
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def predict(self, X):
        X, _ = self._validate_transform_input(X)
        return self.__compute_labels(X, self._centroids)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)










