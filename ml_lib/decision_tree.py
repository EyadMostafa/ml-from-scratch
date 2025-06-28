import numpy as np
from abc import abstractmethod
from .base_model import BaseModel

class DecisionTreeBase(BaseModel):
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features

    @abstractmethod 
    def _compute_impurity(self, left, right):
        pass

    @abstractmethod
    def _best_split(self, X, y):
        pass
    
    @abstractmethod
    def _build_tree(self, X, y):
        pass


class TreeNode:
    def __init__(self,
                 impurity=None, 
                 samples=None,
                 values=None,
                 label=None):
        self._impurity = impurity
        self._samples = samples
        self._values = values
        self._label = label
        self._feature_idx = None
        self._threshold=None
        self._left_node = None
        self._right_node = None

    def _is_leaf(self):
        return self._left_node is None and self._right_node is None


class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__labels = None
        self._root = None

    def _majority_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _compute_impurity(self, values_left, values_right, samples_left, samples_right):
        squared_probs_left = (values_left/samples_left)**2 if samples_left > 0 else 0
        squared_probs_right = (values_right/samples_right)**2 if samples_right > 0 else 0

        left_impurity = 1 - np.sum(squared_probs_left)
        right_impurity = 1 - np.sum(squared_probs_right)

        return left_impurity, right_impurity


    def _best_split(self, X, y):
        _, n = X.shape
        best_X_left = best_y_left = None
        best_X_right = best_y_right = None
        best_threshold = None
        best_feature_idx = None
        best_samples_left = best_samples_right = None
        best_values_left = best_values_right = None
        min_impurity_left = min_impurity_right = None
        min_cost = np.inf

        features_to_consider = np.random.choice(n, self._max_features, replace=False) if self._max_features else range(n)

        for i in features_to_consider:
            feature = X[:, i]
            sorted_idx = np.argsort(feature)
            feature = feature[sorted_idx]
            X_sorted = X[sorted_idx]
            y_sorted = y[sorted_idx]
            thresholds = (feature[1:] + feature[:-1]) / 2

            for threshold in thresholds:
                left_mask = feature <= threshold
                right_mask = feature > threshold
                X_left, y_left = X_sorted[left_mask], y_sorted[left_mask]
                X_right, y_right = X_sorted[right_mask], y_sorted[right_mask]
                samples_left = len(y_left)
                samples_right = len(y_right)
                total_samples = samples_left + samples_right
                values_left = np.array([(y_left == val).sum() for val in self.__labels])
                values_right = np.array([(y_right == val).sum() for val in self.__labels])
                left_impurity, right_impurity = self._compute_impurity(values_left, values_right, 
                                                                       samples_left, samples_right)
                cost = (samples_left / total_samples) * left_impurity + (samples_right / total_samples) * right_impurity 

                if(cost < min_cost):
                    best_X_left, best_y_left = X_left, y_left
                    best_X_right, best_y_right = X_right, y_right
                    best_threshold = threshold
                    best_feature_idx = i
                    best_samples_left, best_samples_right = samples_left, samples_right
                    best_values_left, best_values_right = values_left, values_right
                    min_impurity_left, min_impurity_right = left_impurity, right_impurity
                    min_cost = cost

        if best_y_left is None or best_y_right is None:
            return {
                "feature_idx": None
            }
        
        return {
                "X_left": best_X_left,
                "y_left": best_y_left,
                "X_right": best_X_right,
                "y_right": best_y_right,
                "threshold": best_threshold,
                "feature_idx": best_feature_idx,
                "samples_left": best_samples_left,
                "samples_right": best_samples_right,
                "values_left": best_values_left,
                "values_right": best_values_right,
                "impurity_left": min_impurity_left,
                "impurity_right": min_impurity_right,
                "class_left": self._majority_label(best_y_left), 
                "class_right": self._majority_label(best_y_right),
                "cost": min_cost
               }


    def _build_tree(self, X, y, node, depth=1):

        if (self._max_depth != None and depth >= self._max_depth
           or len(y) < self._min_samples_split
           or len(np.unique(y)) == 1
           or X.shape[1] == 0):
            return None

        split = self._best_split(X, y)

        if (split["feature_idx"] is None
            or split["samples_left"] < self._min_samples_leaf
            or split["samples_right"] < self._min_samples_leaf):
            return node

        node._feature_idx = split["feature_idx"]
        node._threshold = split["threshold"]
        node._left_node = TreeNode(
            impurity=split["impurity_left"],
            samples=split["samples_left"],
            values=split["values_left"],
            label=split["class_left"]
        )
        node._right_node = TreeNode(
            impurity=split["impurity_right"],
            samples=split["samples_right"],
            values=split["values_right"],
            label=split["class_right"]
        )

        self._build_tree(split["X_left"], split["y_left"], node._left_node, depth+1)
        self._build_tree(split["X_right"], split["y_right"], node._right_node, depth+1)

        return node
        
    
    def fit(self, X, y):
        X, y = self._validate_transform_input(X, y)
        self.__labels = np.unique(y)
        self._root = TreeNode(
            impurity=1.0,
            samples=len(y),
            values=np.array([(y == val).sum() for val in self.__labels]),
            label=self._majority_label(y)
        )
        self._build_tree(X, y, self._root)


    def _traverse(self, x, node):
        if node._is_leaf():
            return node._label
        if x[node._feature_idx] <= node._threshold:
            return self._traverse(x, node._left_node)
        else:
            return self._traverse(x, node._right_node)
        
    def predict(self, X):
        X, _ = self._validate_transform_input(X)
        return np.array([self._traverse(x, self._root) for x in X])




