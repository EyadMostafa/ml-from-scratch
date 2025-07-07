import numpy as np
from scipy.stats import multivariate_normal
from .kmeans import KMeans
from utils.helpers import validate_transform_input

class GaussianMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self._n_components = n_components
        self._max_iter = max_iter
        self._tol = tol
        self._weights = []
        self._means = []
        self._covariances = []
        self._log_likelihood = None

    _validate_transform_input = staticmethod(validate_transform_input)

    def __initialize_params(self, X):
        K = self._n_components
        kmeans = KMeans(n_clusters=self._n_components).fit(X)

        for k in range(K):
            mask = kmeans.get_params['labels'] == k
            X_k = X[mask]

            weight = len(X_k) / len(X)
            mean = kmeans.get_params['centroids'][k]
            
            if X_k.shape[0] > 1:
                covariance = np.cov(X_k, rowvar=False)
            else:
                covariance = np.eye(X.shape[1]) * 1e-6
            
            self._weights.append(weight)
            self._means.append(mean)
            self._covariances.append(covariance)

    def __compute_log_likelihood(self, X, weights, means, covariances):
        N = X.shape[0]
        K = self._n_components
        joint_probs = np.zeros((N, K))
        for k in range(K):
            joint_probs[:, k] = self.__compute_joint_probability(X, weights[k], means[k], covariances[k])

        return np.sum(np.log(np.sum(joint_probs, axis=1) + 1e-10))

    def __compute_joint_probability(self, X, weight, mean, covariance):
        dist = multivariate_normal(mean=mean, cov=covariance)
        pdfs = dist.pdf(X)
        return weight * pdfs
    
    def __expectation_step(self, X):
        N = X.shape[0]
        K = self._n_components

        joint_probs = np.zeros((N, K))

        for k in range(K):
            joint_probs[:, k] = self.__compute_joint_probability(
                X=X,
                weight=self._weights[k],
                mean=self._means[k],
                covariance=self._covariances[k]
            )

        posterior_probs = joint_probs / np.sum(joint_probs, axis=1, keepdims=True)

        return np.array(posterior_probs)

    def __maximization_step(self, X):
        N, D = X.shape
        K = self._n_components

        for _ in range(self._max_iter):
            posterior_probs = self.__expectation_step(X)
            new_weights = np.zeros(K)
            new_means = np.zeros((K, D))
            new_covariances = np.zeros((K, D, D))

            for k in range(K):
                gamma_k = posterior_probs[:, k]
                N_k = np.sum(gamma_k)

                new_weights[k] = N_k / N
                new_means[k] = np.sum(gamma_k[:, np.newaxis] * X, axis=0) / N_k

                diff = X - new_means[k]
                weighted_outer = gamma_k[:, np.newaxis, np.newaxis] * np.einsum("ni,nj->nij", diff, diff)
                new_covariances[k] = np.sum(weighted_outer, axis=0) / N_k

            new_log_likelihood = self.__compute_log_likelihood(X, new_weights, new_means, new_covariances)

            if self._log_likelihood is None:
                self._log_likelihood = new_log_likelihood

            log_likelihood_diff = new_log_likelihood - self._log_likelihood 

            if abs(log_likelihood_diff) > self._tol:
                self._weights = new_weights
                self._means = new_means
                self._covariances = new_covariances
                self._posterior_probs = posterior_probs
                self._log_likelihood = new_log_likelihood
            else: break

    def fit(self, X):
        X, _ = self._validate_transform_input(X)

        self.__initialize_params(X)
        self.__maximization_step(X)

        self._weights = np.array(self._weights)
        self._means = np.array(self._means)
        self._covariances = np.array(self._covariances)

    def predict_proba(self, X):
        X, _ = self._validate_transform_input(X)
        return self.__expectation_step(X)
    
    def predict(self, X):
        posterior_probs = self.predict_proba(X)
        return np.argmax(posterior_probs, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)