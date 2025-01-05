# https://en.wikipedia.org/wiki/Naive_Bayes_classifier

import numpy as np
from typing import Self

class NaiveBayes:
    def __init__(self) -> None:
        """
        Initialise the Naive Bayes model.
        """
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Naive Bayes model (effectivly just store the training data for when it is needed).

        Args:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # store class labels
        self.classes = sorted(list(np.unique(y_train)))

        # store the training data and labels
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

        # get number of sample in train and test data
        n_train = self.X_train.shape[0]
        n_test = X_test.shape[0]

        # create an empty array for storing probabilities for each class
        class_posteriors = np.empty((n_test, len(self.classes)))

        # loop through the classes
        for idx, class_label in enumerate(self.classes):

            # filter training data to get only samples from the current class
            class_data = self.X_train[self.y_train == class_label]
            
            # get the class mean and standard deviation of the class for each feature
            class_mean = np.mean(class_data, axis=0)
            class_std = np.std(class_data, axis=0)

            # calculate the prior probability as the fraction of the class in the training data
            prior = class_data.shape[0]/n_train

            # calculate the likelihood, i.e. the condition probability of observing the test value for each feature
            # an assumption is made here that features are independant and normally distributed
            likelihood = np.prod(1/(2*np.pi*class_std**2)*np.exp(-(X_test-class_mean)**2/(2*class_std**2)), axis=1)

            # the posterior for the class is the product of the prior and likelihood
            # (we can ignore division by the evidence for now since this is simply a constant scaling factor)
            class_posteriors[:, idx] = prior * likelihood
        
        # scale the posteriors to represent true probabilities (i.e. sum to 1 across all classes)
        pred_probs = class_posteriors/np.sum(class_posteriors, axis=1)[:, None]

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
        y_pred = np.argmax(pred_probs, axis=1)

        # convert the indices back to the original class labels
        y_pred = np.array(self.classes)[y_pred]

        return y_pred