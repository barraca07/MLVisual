import numpy as np
from typing import Self
from utils import map_class_labels

class DecisionStump:
    """
    A class representing a decision stump (i.e. a decision tree with just a single split).
    """

    def __init__(self) -> None:
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> tuple[Self, float]:
        """
        Fit the Decision Stump by minimising the weighted sum of misclassified points.

        Parameters:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).
            weights (np.ndarray): The sample weights of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # initialise the min weighted error
        min_weighted_error = np.inf
        
        # loop through each feature in the training data
        for feature_idx in range(X_train.shape[1]):

            # get the min and max of the current feature
            feature_max = np.max(X_train[:, feature_idx])
            feature_min = np.min(X_train[:, feature_idx])
            
            # create an array of equally spaced values in the feature's range as possible values to split the node
            split_values = np.linspace(feature_min, feature_max, 200)
            
            # loop through the values
            for value in split_values:

                # split the weights and target values by the current split condition
                condition = X_train[:,feature_idx]<=value
                y_pred_0 = y_train[condition]
                weights_0 = weights[condition]
                y_pred_1 = y_train[~condition]
                weights_1 = weights[~condition]

                # check both less than and greater than the split value being each class
                for multiplier in [1, -1]:

                    # calculate the sum of the weighted errors for misclassified points
                    weighted_error = np.sum(weights_0[y_pred_0 != -multiplier]) + np.sum(weights_1[y_pred_1 != multiplier])
                    
                    # check if current error is less than previous min
                    if weighted_error < min_weighted_error:

                        # save new minimum and split parameters
                        min_weighted_error = weighted_error
                        self.feature_idx = feature_idx
                        self.value = value
                        self.less = True if multiplier == 1 else False

        return self, min_weighted_error

    def predict(self, X_test: np.ndarray):
        # initialise the predictions as all ones
        y_pred = np.ones(X_test.shape[0])

        # make the prediction for the class based on the previously saved best split parameters
        if self.less:
            minus_ones = X_test[:, self.feature_idx] <= self.value
        else:
            minus_ones = X_test[:, self.feature_idx] > self.value
        
        # set the minus one predictions
        y_pred[minus_ones] = -1

        return y_pred

class AdaBoost:
    def __init__(self, num_learners: int=30) -> None:
        self.num_learners = num_learners

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Initialise the AdaBoost model.

        Parameters:
            learning_rate (float): The gradient descent step size.
            tolerance (float): The tolerance for gradient norm to stop training.
            max_itr (int): Maximum number of iterations for gradient descent.
        """

        # re-label the classes to real numbers 0, 1, etc.
        y_train, classes = map_class_labels(y_train)
        if len(classes) != 2:
            raise Exception("The target vector y_train must contain exactly two classes.")
        
        # store original class labels
        self.classes = classes

        # change zero class label to be -1
        y_train[y_train == 0] = -1

        # get number of samples
        n = X_train.shape[0]

        # initialise sample weights
        weights = np.ones(n)/n

        # initialise an empty list to store the ensemble of weak classifiers
        ensemble = []

        # loop once for each weak learner to be added to the ensemble
        for _ in range(self.num_learners):

            # fit the decision stump based on the current weights
            stump, weighted_error = DecisionStump().fit(X_train, y_train, weights)

            # calculate alpha from the misclassified weighted error sum
            alpha = 0.5 * np.log((1-weighted_error)/weighted_error) if weighted_error != 0 else np.inf

            # add the stump and alpha value to the ensemble
            ensemble.append((stump, alpha))

            if weighted_error == 0:
                break

            # update the weights, increasing those previously misclassified 
            weights = weights * np.exp(-y_train*alpha*stump.predict(X_train))

            # normalise the weights so they sum to 1
            weights = weights/np.sum(weights)



        # store the ensemble
        self.ensemble = ensemble

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """

        # initialise an output vector of zeros
        output = np.zeros(X_test.shape[0])

        # loop through each stump in the ensemble
        for stump, alpha in self.ensemble:

            # add the prediction (scaled by alpha) to the current output
            output += stump.predict(X_test)*alpha

        # convert output to probability
        pred_prob_1 = (1/(1+np.exp(-2*output)))[:, None]

        # construct the probability matrix for both classes
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

        # get the class index with the max probability for each sample
        index_max = np.argmax(pred_probs, axis=1)

        # convert the indices back to the original class labels
        y_pred = np.array(self.classes)[index_max]

        return y_pred