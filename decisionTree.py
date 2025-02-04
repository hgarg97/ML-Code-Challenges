# https://www.youtube.com/watch?v=NxEHSAfFlK8
# https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch

import numpy as np
from collections import Counter
import math

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        # Stopping Criteria
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        # Helper Function to Grow Tree
        self.root = self._grow_tree(X, y)



    def _grow_tree(self, X, y, depth = 0):

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check for the Stopping Criteria
        if (depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the Best Split
        feat_idx = np.random.choice(n_feats, self.n_features, replace = False)

        best_features, best_threshold = self._best_split(X, y, feat_idx)

        # Create Child Nodes
        left_idxs, right_idxs = self._split(X[:, best_features], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_features, best_threshold, left, right)


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]

        return value
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate Information Gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    

    def _information_gain(self, y, X_column, threshold):
        # Parent Entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the Weighted Average Entropy of Children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate the information Gain
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()

        return left_idxs, right_idxs

    def _gini(self, y):
        counts = Counter(y)
        total = len(y)
        gini = 1.0
        for count in counts.value():
            p = count/total
            gini -= p**2
        return gini


    def _entropy(self, y):
        counts = Counter(y)
        total = len(y)
        ent = 0.0
        for count in counts.values():
            p = count / total
            if p>0:
                ent -= p * math.log2(p)
        return ent
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)