import numpy as np
from .base_model import BaseModel
from scipy.stats import mode
from .decision_tree import DecisionTreeClassifier
import copy

class BaggingClassifier(BaseModel):
    """
    Bagging (Bootstrap Aggregating) ensemble classifier.

    This class trains multiple copies of a base estimator on random subsets
    of the training data and aggregates their predictions via majority voting.

    Parameters:
        estimator (object): Base estimator to fit on random subsets. Must implement fit and predict.
        n_estimators (int): Number of base estimators to train.
        max_samples (float or int): Fraction or number of samples to draw for each base estimator.
        max_features (float or int): Fraction or number of features to draw for each base estimator.
        bootstrap (bool): Whether samples are drawn with replacement (bootstrap) or without.
    """
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True):
        self.__estimator = estimator
        self.__n_estimators = n_estimators
        self.__max_samples = max_samples
        self.__max_features = max_features
        self.__bootstrap = bootstrap
        self.__trained_estimators = []


    def __generate_random_samples(self, m):
        """
        Generate random indices for sample subset selection.

        Args:
            m (int): Total number of samples.

        Returns:
            np.ndarray: Array of selected row indices.
        """
        return np.random.choice(m, self.__max_samples, replace=self.__bootstrap)
    
    def __generate_random_features(self, n):
        """
        Generate random indices for feature subset selection.

        Args:
            n (int): Total number of features.

        Returns:
            np.ndarray: Array of selected feature indices.
        """
        return np.random.choice(n, self.__max_features, replace=False)
    
    def __train_estimators(self, X, y):
        """
        Train all base estimators on random subsets of data and features.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        for _ in range(self.__n_estimators):
            subset_idx = self.__generate_random_samples(X.shape[0])
            features_idx = self.__generate_random_features(X.shape[1])

            X_subset = X[np.ix_(subset_idx, features_idx)]
            y_subset = y[subset_idx]

            estimator = copy.deepcopy(self.__estimator)
            estimator.fit(X_subset, y_subset)
            self.__trained_estimators.append((estimator, features_idx))

    def fit(self, X, y):
        """
        Fit the bagging ensemble on the training data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Target labels.
        """
        X, y = self._validate_transform_input(X, y)
        m, n = X.shape

        if self.__estimator is None: self.__estimator = DecisionTreeClassifier()

        if isinstance(self.__max_samples, float): 
            self.__max_samples = int(m * self.__max_samples)
        if self.__max_samples > m: self.__max_samples = m

        if isinstance(self.__max_features, float): 
            self.__max_features = int(n * self.__max_features)
        if self.__max_features > n: self.__max_features = n

        self.__train_estimators(X, y)

    def predict(self, X):
        """
        Predict class labels for the input data using majority voting.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X, _ = self._validate_transform_input(X)

        predictions = []

        for estimator, features_idx in self.__trained_estimators:
            predictions.append(estimator.predict(X[:, features_idx]))

        predictions = np.array(predictions)

        return mode(predictions, axis=0).mode.flatten()



