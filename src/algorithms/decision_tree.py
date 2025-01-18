# https://en.wikipedia.org/wiki/Decision_tree_learning

import numpy as np
from typing import Self
from .utils import map_class_labels

class Node:
    """
    A class representing a single node or split in a decision tree
    """

    def __init__(self, probabilities: None | np.ndarray=None):
        """
        Initialise the Node.

        Parameters:
            probabilities (None | np.ndarray): If the node is a leaf node, probabilities is an array of shape (n_classes,) which represents the probabilty of each class. If not a leaf node it is None.
        """

        self.probabilities: None | np.ndarray = probabilities
        self.feature_idx: None | int = None
        self.split_value: None | float = None
        self.left: None | Node = None
        self.right: None | Node = None


class DecisionTree:
    def __init__(self, max_depth: int=2, features_to_use: list[int]=[]) -> None:
        """
        Initialise the Decision Tree model.

        Parameters:
            max_depth (int): The maximum depth, from root to leaf, of the decision tree.
            features_to_use (list[int]): The subset of features on which to consider splitting the data. If empty, all features will be used.
        """

        self.max_depth = max_depth
        self.features_to_use = features_to_use

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Decision Tree by using the recursive `self.split` function.

        Parameters:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # re-label the classes to real numbers 0, 1, etc.
        y_train, classes = map_class_labels(y_train)
        
        # store original class labels
        self.classes = classes

        # get an iterator of the features to use to train the decision tree
        # if self.features_to_use is empty we simply use all available features
        self.features_to_use = list(range(X_train.shape[1])) if len(self.features_to_use) == 0 else self.features_to_use

        # recursively generate the tree
        self.tree = self.split(X_train, y_train)
        return self
    
    def split(self, X: np.ndarray, y: np.ndarray, depth: int=1) -> Node:
        """
        A recursive function for building the decision tree by splitting the data at Nodes until the max depth is reached or all the Nodes are pure.

        Args:
            X (np.ndarray): Training data of shape (k_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            depth (int): The current depth of the decision tree.

        Returns:
            Node: The current node.
        """

        # break the recursion if max depth is reached or the target doesn't contain at least two classes
        if depth > self.max_depth or len(set(y)) < 2:
            return Node(probabilities=np.bincount(y, minlength=len(self.classes))/y.shape[0])
        
        # initialise the best split parameters
        best_split_information_gain = -np.inf
        best_split_feature = None
        best_split_value = None
        
        # loop through each features
        for feature_idx in self.features_to_use:

            # get the min and max of the current feature
            feature_max = np.max(X[:, feature_idx])
            feature_min = np.min(X[:, feature_idx])

            # create an array of equally spaced values in the feature's range as possible values to split the node
            split_values = np.linspace(feature_min, feature_max, 100)

            # loop through the values
            for value in split_values:

                # split the target vector according to the value
                condition = X[:,feature_idx]<=value
                y_1 = y[condition]
                y_2 = y[~condition]

                # calculate the information gain from the split
                information_gain = self.information_gain(y_1, y_2)

                # if the information gain is greater than the current best split it becomes the best split
                if information_gain > best_split_information_gain:
                    best_split_information_gain = information_gain
                    best_split_feature = feature_idx
                    best_split_value = value
        
        # create an empty node
        curr_node = Node()

        # the best split condition found previously
        condition = X[:, best_split_feature] <= best_split_value

        # if the split results in either new target vector being empty break from the recursion
        y_left = y[condition]
        y_right = y[~condition]
        if y_left.size == 0 or y_right.size == 0:
            return Node(probabilities=np.bincount(y, minlength=len(self.classes))/y.shape[0])
        
        # recursively split the newly created parts
        curr_node.left = self.split(X[condition], y[condition], depth=depth+1)
        curr_node.right = self.split(X[~condition], y[~condition], depth=depth+1)

        # save the feature and value used for the split
        curr_node.feature_idx = best_split_feature
        curr_node.split_value = best_split_value

        # return the current node
        return curr_node

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        
        # initialise an empty list for the predictions
        predictions = []
        # loop through each test sample
        for i in range(X_test.shape[0]):

            # get the current sample
            X_curr = X_test[i, :]

            # start at the root of the decision tree
            curr_node = self.tree
            
            # traverse the decision tree until a leaf node is reached
            while curr_node.probabilities is None:
                if X_curr[curr_node.feature_idx] < curr_node.split_value:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            
            # add the predicted probabilities to the output list
            predictions.append(curr_node.probabilities)

        # convert the ouput list to an array
        y_pred = np.array(predictions)

        return y_pred
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """

        # get the probability predictions
        pred_probs = self.predict_proba(X_test)

        # get the class index with the max probability for each sample
        index_max = np.argmax(pred_probs, axis=1)

        # convert the indices back to the original class labels
        y_pred = np.array(self.classes)[index_max]

        return y_pred
    
    def entropy(self, y: np.ndarray) -> float:
        """
        A utility function for calculating the entropy of an array of target class labels

        Args:
            y (np.ndarray): Target values of shape (k_samples,)

        Returns:
            float: The entropy of the list.
        """
        
        # get the length of the array
        n = y.shape[0]
        
        # get the counts of each class in the array
        counts = np.bincount(y)
        counts = counts[counts > 0]

        # get the probability of each class
        prob = counts/n

        # compute the entropy
        entropy = np.sum(-prob * np.log2(prob))

        return entropy

    def information_gain(self, y_1: np.ndarray, y_2: np.ndarray) -> float:
        """
        A utility function for calculating the information gain associated with a given split of the data.

        Args:
            y_1 (np.ndarray): First part of target values of shape (k1_samples,)
            y_2 (np.ndarray): Second part of target values of shape (k2_samples,)

        Returns:
            float: The calculated information gain of the split.
        """
        
        # get the length of each array and total number
        n_1 = y_1.shape[0]
        n_2 = y_2.shape[0]
        n = n_1 + n_2

        # concatenate the arrays
        combined_labels = np.concatenate([y_1, y_2])

        # calculate the information gain
        information_gain = self.entropy(combined_labels) - n_1/n * self.entropy(y_1) - n_2/n * self.entropy(y_2)

        return information_gain