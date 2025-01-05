# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

import numpy as np
from typing import Self
from utils import map_class_labels

class NearestNeighbours:
    def __init__(self, k: int=9) -> None:
        """
        Initialise the KNN model.

        Args:
            k (int): The number of neighbours to consider for each test point.
        """

        self.k = k

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the KNN model (effectivly just store the training data for when it is needed).

        Args:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """
        
        # store the unique class labels
        self.classes = sorted(list(np.unique(y_train)))

        # store the training data
        self.X_train = X_train
        self.y_train = y_train

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """

        # calculate the euclidian distance between every combination of train and test points
        distances = np.sqrt(np.sum((X_test[:, None, :] - self.X_train[None, :, :]) ** 2, axis=2))

        # get the indices of the nearest k points
        indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # get the classes of the nearest k points
        nearest_k_classes = self.y_train[indices]

        class_probabilities = []
        # loop through the classes
        for label in self.classes:

            # the probability for a class is the number of occurances in the k-nearest divided by k
            class_prob = np.sum(nearest_k_classes == label, axis=1)/self.k

            # add the class probability to output list
            class_probabilities.append(class_prob[:, None])

        # convert list to array
        pred_probs = np.hstack(class_probabilities)

        return pred_probs
    
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

        # get the index of the class with the max probability
        index_max = np.argmax(pred_probs, axis=1)

        # convert the indices back to the original class labels
        y_pred = np.array(self.classes)[index_max]

        return y_pred