import numpy as np
from .base_model import BaseModel
from .decision_tree import DecisionTreeClassifier
from copy import deepcopy


class AdaBoostClassifier(BaseModel):
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0):
        self.__estimator = estimator
        self.__n_estimators = n_estimators
        self.__learning_rate = learning_rate
        self.__w = None
        self.__trained_estimators = None


    def __update_weights(self, y, preds):
        r = np.sum(self.__w * (y != preds)) / np.sum(self.__w)
        r = np.clip(r, 1e-10, 1 - 1e-10)

        estimator_weight = self.__learning_rate * 0.5 * np.log((1-r)/r)
        self.__w *= np.exp(estimator_weight * (y != preds).astype(float))
        self.__w /= np.sum(self.__w)

        return estimator_weight

    def __train_estimators(self, X, y):
        for _ in range(self.__n_estimators):
            estimator = deepcopy(self.__estimator)
 
            m_samples = len(X)
            indices = np.random.choice(
                np.arange(m_samples),
                size=m_samples,
                replace=True,
                p=self.__w
            )

            X_resampled = X[indices]
            y_resampled = y[indices]

            estimator.fit(X_resampled,y_resampled)
            preds = estimator.predict(X)
            estimator_weight = self.__update_weights(y, preds)

            self.__trained_estimators.append((estimator, estimator_weight))

    def fit(self, X, y):
        X, y = self._validate_transform_input(X,y)  
        m = len(X)
        self.__w = np.full(shape=m, fill_value=1/m)
        self.__trained_estimators = []

        if self.__estimator is None: self.__estimator = DecisionTreeClassifier(max_depth=1)
        self.__train_estimators(X, y)

    def predict(self, X):
        if len(self.__trained_estimators) == 0:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self._validate_transform_input(X)

        weighted_sum = np.zeros(len(X))

        for estimator, weight in self.__trained_estimators:
            pred = estimator.predict(X)

            pred = np.where(pred == 0, -1, 1)

            weighted_sum += weight * pred

        return np.where(weighted_sum >= 0, 1, 0)

        

        






