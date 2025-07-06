import numpy as np
from collections import deque
from utils.helpers import validate_transform_input

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self._eps = eps
        self._min_samples = min_samples
        self._labels = None
        self._components_idxs = []
        self._components = []
        self.__vis = None
        self.__candidate_queue = deque()

    _validate_transform_input = staticmethod(validate_transform_input)

    def __compute_euclidean_dist(self, X, component):
        return np.linalg.norm(X - component, axis=1)
    
    def __try_start_cluster(self, X, idx, label):
            candidate = X[idx]
            distances = self.__compute_euclidean_dist(X, candidate)
            neighbour_idxs = np.where(distances <= self._eps)[0]

            if len(neighbour_idxs) < self._min_samples:
                return False
            
            self._components_idxs.append(idx)
            self._components.append(candidate)
            self.__candidate_queue.extend(neighbour_idxs)
            self._labels[idx] = label
            self._labels[neighbour_idxs] = label
            self.__vis[idx] = True

            return True
            
    def __expand_cluster(self, X):
         while self.__candidate_queue:
             candidate_idx = self.__candidate_queue.popleft()
             if self.__vis[candidate_idx]: 
                 continue

             candidate = X[candidate_idx]
             self.__vis[candidate_idx] = True

             distances = self.__compute_euclidean_dist(X, candidate)
             neighbour_idxs = np.where(distances <= self._eps)[0]

             if len(neighbour_idxs) >= self._min_samples:
                 self._components_idxs.append(candidate_idx)
                 self._components.append(candidate)
                 for i in neighbour_idxs:
                      if (not self.__vis[i]) and (i not in self.__candidate_queue):
                           self.__candidate_queue.append(i)

             self._labels[neighbour_idxs] = self._labels[candidate_idx]
             
    def fit(self, X):
        X, _ = self._validate_transform_input(X)
        m = X.shape[0]
        self._labels = np.full(m, -1)
        self.__vis = np.zeros(m, dtype=bool)
        cluster_count = 0

        for i in range(m):
              if self.__vis[i]: continue
              if self.__try_start_cluster(X, i, cluster_count):
                   self.__expand_cluster(X)
                   cluster_count += 1

        self._components_idxs = np.array(self._components_idxs)
        self._components = np.array(self._components)
    
    def fit_predict(self, X):
         self.fit(X)
         return self._labels









