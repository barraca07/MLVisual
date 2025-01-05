# https://en.wikipedia.org/wiki/Random_forest

import numpy as np
from typing import Self
from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, num_trees: int=10, max_depth: int=2, samples_fraction: float=0.5, features_fraction: float=0.5) -> None:
        """
        Initialise the Random Forest model.

        Parameters:
            num_trees (int): The number of trees in the forest.
            max_depth (int): The depth of each tree.
            samples_fraction (float): The fraction of samples to use when training each tree.
            features_fraction (float): The fraction of features to use when training each tree.
        """

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Random Forest model.

        Parameters:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # store the list of unique classes
        self.classes = sorted(list(set(y_train)))

        # loop through the number of trees to create
        self.trees = []
        for _ in range(self.num_trees):

            # get a subset of samples and features on which to train the current tree
            samples_idx = np.random.choice(np.arange(y_train.shape[0]), np.ceil(y_train.shape[0]*self.samples_fraction).astype(int), replace=False)
            features_idx = np.random.choice(np.arange(X_train.shape[1]), np.ceil(X_train.shape[1]*self.features_fraction).astype(int), replace=False)
            X_sample = X_train[samples_idx]
            y_sample = y_train[samples_idx]

            # train a decision tree on the sample using just the feature subset
            curr_tree = DecisionTree(max_depth=self.max_depth, features_to_use=features_idx).fit(X_sample, y_sample)
            
            # add the tree to the ensemble
            self.trees.append(curr_tree)
        
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
                
        preds = []

        # loop through each tree
        for tree in self.trees:

            # initialise true probs array accounting for the fact that each tree may not contain samples from every class
            true_probs = np.zeros((X_test.shape[0], len(self.classes)))
            
            # get the prediction for the current tree
            probs = tree.predict_proba(X_test)

            # get mapping of tree class labels to index
            class_to_subset_idx = {label: idx for idx, label in enumerate(tree.classes)}
            
            # loop through each class
            for i, label in enumerate(self.classes):

                # if class in in the tree
                if label in class_to_subset_idx:

                    # get the index of the class for the tree
                    subset_index = class_to_subset_idx[label]

                    # update the correct column with the predicted probabilities for that class
                    true_probs[:, i] = probs[:, subset_index]
            
            # add the output to the list of predictions
            preds.append(true_probs)

        # average the predictions across all trees
        y_pred = np.mean(np.array(preds), axis=0)

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