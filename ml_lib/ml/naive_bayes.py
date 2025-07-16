import numpy as np
from .base_model import BaseModel

# Untested...
# support class_prior param?

class MultinomialNB(BaseModel):
    def __init__(self, alpha=1, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_log_prior = None
        self.feature_log_prior = None
        self.class_count = None
        self.feature_count = None

    def __compute_class_counts(self, y):
        values, counts = np.unique(y, return_counts=True)
        return dict(zip(values, counts))

    def __compute_feature_counts(self, X, y):
        feature_counts = dict()

        for val in self.class_count.keys():
            X_filtered = X[y == val]
            feature_sum = np.sum(X_filtered, axis=0)
            feature_counts[val] = feature_sum

        return feature_counts

    def __compute_class_log_prior(self, counts):
       total = np.sum(list(counts.values()))
       return {k: np.log(v / total) for k, v in counts.items()}

    def __compute_feature_log_prior(self, counts):
        return {k: np.log((v + self.alpha) / np.sum(v + self.alpha)) for k, v in counts.items()}

    def fit(self, X, y):
        X, y = self._validate_transform_input(X, y)

        self.class_count = self.__compute_class_counts(y)
        self.feature_count = self.__compute_feature_counts(X, y)

        if self.fit_prior:
            self.class_log_prior = self.__compute_class_log_prior(self.class_count)
        else: 
            keys = np.unique(y)
            vals = np.ones(len(keys)) / len(keys)
            self.class_log_prior = dict(zip(keys, np.log(vals)))

        self.feature_log_prior = self.__compute_feature_log_prior(self.feature_count)

        return self

    def predict_proba(self, X):
        X, _ = self._validate_transform_input(X)
        probs = []

        for x in X:
            log_scores = []
            for c in self.class_count:
                features_log_prior = self.feature_log_prior[c]
                log_scores.append(self.class_log_prior[c] + (x @ features_log_prior)) 
            log_scores = np.array(log_scores)
            log_scores = log_scores - np.max(log_scores)
            probs.append(np.exp(log_scores) / np.sum(np.exp(log_scores)))

        return np.array(probs)  

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)