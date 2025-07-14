import numpy as np
from abc import abstractmethod
from .base_model import BaseModel

class TreeNode:
    """
    A node in the decision tree.

    Stores information about a decision or leaf node, including impurity,
    sample count, predicted label, split criteria, and child nodes.

    Attributes:
        _impurity (float): Impurity of the node (e.g., Gini index).
        _samples (int): Number of samples at the node.
        _values (ndarray): Class distribution at the node.
        _label (any): Predicted label (majority class) at the node.
        _feature_idx (int): Index of the feature used for the split.
        _threshold (float): Threshold value for the split.
        _left_node (TreeNode): Left child node.
        _right_node (TreeNode): Right child node.
    """
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
        """Check if the current node is a leaf (no children)."""
        return self._left_node is None and self._right_node is None
    

class DecisionTreeBase(BaseModel):
    """
    Abstract base class for decision tree implementations.

    Provides core configuration and interface for building decision trees.

    Parameters:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split a node.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        max_features (int): Number of features to consider when looking for the best split.
    """
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
    def _compute_impurity(self, values_left, values_right, samples_left, samples_right):
        pass

    @abstractmethod
    def _best_split(self, X, y):
        pass
    
    @abstractmethod
    def _build_tree(self, X, y, node, depth=0):
        pass


class DecisionTreeClassifier(DecisionTreeBase):
    """
    Decision tree classifier using the CART algorithm.

    Builds a binary decision tree for classification tasks based on
    Gini impurity and recursively splits nodes to minimize impurity.

    Inherits from DecisionTreeBase.

    Attributes:
        _labels (ndarray): Unique class labels in the training data.
        __root (TreeNode): Root node of the constructed decision tree.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._labels = None
        self.__root = None

    def _majority_label(self, y):
        """
        Determine the majority class label in the given array.

        Parameters:
            y (ndarray): Array of class labels.

        Returns:
            The most frequent label.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _compute_impurity(self, values_left, values_right, samples_left, samples_right):
        """
        Compute the Gini impurity for the left and right child nodes.

        Parameters:
            values_left (ndarray): Class distribution in the left split.
            values_right (ndarray): Class distribution in the right split.
            samples_left (int): Number of samples in the left split.
            samples_right (int): Number of samples in the right split.

        Returns:
            tuple: (left_impurity, right_impurity)
        """
        squared_probs_left = (values_left/samples_left)**2 if samples_left > 0 else 0
        squared_probs_right = (values_right/samples_right)**2 if samples_right > 0 else 0

        left_impurity = np.round(1 - np.sum(squared_probs_left), 3)
        right_impurity = np.round(1 - np.sum(squared_probs_right), 3)

        return left_impurity, right_impurity


    def _best_split(self, X, y):
        """
        Identify the best feature and threshold to split the data to reduce impurity.

        Parameters:
            X (ndarray): Feature matrix.
            y (ndarray): Target vector.

        Returns:
            dict: Dictionary containing the best split's details, including:
                - X_left, X_right: Subsets of X.
                - y_left, y_right: Subsets of y.
                - threshold: Chosen threshold value.
                - feature_idx: Index of the best feature.
                - samples_left, samples_right: Sample counts.
                - values_left, values_right: Class distributions.
                - impurity_left, impurity_right: Gini impurities.
                - class_left, class_right: Majority class in each split.
                - cost: Weighted impurity of the split.
        """
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
                values_left = np.array([(y_left == val).sum() for val in self._labels])
                values_right = np.array([(y_right == val).sum() for val in self._labels])
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


    def _build_tree(self, X, y, node, depth=0):
        """
        Recursively grow the decision tree by choosing the best splits.

        Parameters:
            X (ndarray): Feature matrix at current node.
            y (ndarray): Label vector at current node.
            node (TreeNode): Current node in the tree.
            depth (int): Current depth of the node.

        Returns:
            TreeNode: The constructed subtree.
        """

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
        """
        Fit the decision tree classifier on training data.

        Parameters:
            X (ndarray): Training feature matrix.
            y (ndarray): Training label vector.
        """
        X, y = self._validate_transform_input(X, y)
        self._labels = np.unique(y)

        values = np.array([(y == val).sum() for val in self._labels])
        initial_impurity, _ = self._compute_impurity(values, [0,0,0], y.shape[0], 0)
        self.__root = TreeNode(
            impurity=initial_impurity,
            samples=len(y),
            values=np.array([(y == val).sum() for val in self._labels]),
            label=self._majority_label(y)
        )
        self._build_tree(X, y, self.__root)


    def _traverse(self, x, node):
        """
        Traverse the tree recursively to predict the label for a single sample.

        Parameters:
            x (ndarray): Input sample.
            node (TreeNode): Current node to evaluate.

        Returns:
            Predicted label.
        """
        if node._is_leaf():
            return node._label
        if x[node._feature_idx] <= node._threshold:
            return self._traverse(x, node._left_node)
        else:
            return self._traverse(x, node._right_node)
        
    def predict(self, X):
        """
        Predict class labels for the input samples.

        Parameters:
            X (ndarray): Input feature matrix.

        Returns:
            ndarray: Predicted class labels.
        """
        if not self.__root:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X, _ = self._validate_transform_input(X)
        return np.array([self._traverse(x, self.__root) for x in X])




