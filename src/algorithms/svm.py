import numpy as np
from typing import Self
from algorithms import LogisticRegression
from utils import map_class_labels

class SupportVectorMachine:
    def __init__(self, C: float=0.01, learning_rate: float=5e-2, tolerance: float=1e-5, max_itr: int=10000) -> None:
        """
        Initialise the Support Vector Machine model.

        Parameters:
            C (float): The C hyperparameter for SVM classification.
            learning_rate (float): The gradient descent step size.
            tolerance (float): The tolerance for gradient norm to stop training.
            max_itr (int): Maximum number of iterations for gradient descent.
        """

        self.C = C
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_itr = int(max_itr)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self: 
        """
        Fit the Support Vector Machine model using gradient descent.

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

        # change zero class label to be -1
        y_train[y_train == 0] = -1

        # add a column of 1s to X for the intercept coefficient
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        # initialise model parameters
        weights = np.zeros(X_train.shape[1])

        # enter gradient descent loop for up to the max number of iterations
        # break when the gradient norm is less than the tolerance
        for i in range(self.max_itr):

            # calculate the gradient
            grad_zero_indices = y_train * (X_train @ weights) > 1
            grad = (X_train.T * y_train).T
            grad[grad_zero_indices] = 0
            grad = np.concatenate(([0.0], weights[1:])) - self.C*np.sum(grad, axis=0)

            # perform parameter update step
            weights = weights - self.learning_rate * grad

            if np.linalg.norm(grad) < self.tolerance:
                break
        
        self.weights = weights

        # fit a logistic regression model to convert the scores into probabilities
        train_preds = (X_train @ self.weights).reshape(-1, 1)
        y_train[y_train == -1] = 0
        self.Platt_scaling = LogisticRegression().fit(train_preds, y_train)

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

        pred_prob_1 = self.Platt_scaling.predict_proba((X_test @ self.weights).reshape(-1, 1))[:, 1][:, None]

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

        # add a column of 1s to X for the intercept coefficient
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        # make the class predictions
        y_pred = np.sign(X_test @ self.weights).astype(int)

        # change -1 class label to be 0
        y_pred[y_pred == -1] = 0

        # convert back to the original class labels
        y_pred = np.array(self.classes)[y_pred]

        return y_pred