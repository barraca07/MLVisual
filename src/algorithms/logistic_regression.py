# https://en.wikipedia.org/wiki/Logistic_regression

import numpy as np
from typing import Self
from utils import map_class_labels

class LogisticRegression:
    def __init__(self, learning_rate: float=0.01, tolerance: float=1e-4, max_itr: int=10000) -> None:
        """
        Initialise the Logistic Regression model.

        Parameters:
            learning_rate (float): The gradient descent step size.
            tolerance (float): The tolerance for gradient norm to stop training.
            max_itr (int): Maximum number of iterations for gradient descent.
        """

        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_itr = max_itr

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Logistic Regression model using gradient descent.

        Parameters:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # re-label the classes to real numbers 0, 1, etc.
        y_train, classes = map_class_labels(y_train)
        if len(classes) != 2:
            raise Exception("The target vector y_train must contain exactly two classes.")
        
        # store original class labels
        self.classes = classes

        # get the number of samples and features
        n, m = X_train.shape
        
        # add a column of 1s to X for the intercept coefficient
        X_train = np.hstack((np.ones((n, 1)), X_train))

        # initialise parameter vector
        beta = np.ones(m+1)

        # enter gradient descent loop for up to the max number of iterations
        # break when the gradient norm is less than the tolerance
        for _ in range(self.max_itr):

            # calculate the gradient
            errors = 1/(1 + np.exp(-X_train @ beta)) - y_train
            grad = 1/n * X_train.T @ errors
            
            # perform beta update step
            beta = beta - self.learning_rate * grad

            # break if tolerance is reached
            if np.linalg.norm(grad, 2) < self.tolerance:
                break
        
        # store parameters
        self.beta = beta

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """

        # add a column of 1s to X for the intercept coefficient
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        # calculate the probability of the second class
        pred_prob_1 = 1/(1 + np.exp(-(X_test @ self.beta)))[:, None]

        # construct the matrix for probability of both classes
        pred_probs = np.hstack([1-pred_prob_1, pred_prob_1])

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