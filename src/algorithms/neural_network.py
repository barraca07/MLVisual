import numpy as np
from typing import Self, Callable
from .utils import map_class_labels

class NeuralNetwork:
    def __init__(self, hidden_shape: list[int]=[8, 8], learning_rate: float= 1e-2, epochs: int= 10) -> None:
        """
        Initialise a fully connected Neural Network classification model.

        Parameters:
            hidden_shape (list[int]): The shape of the hidded layers in the network.
            learning_rate (float): The backpropagation step size.
            epochs (float): The number of epochs to train the neural network for.
        """

        self.hidden_shape=hidden_shape
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Fit the Neural Network using backpropagation.

        Parameters:
            X_train (np.ndarray): Training data of shape (n_samples, n_features).
            y_train (np.ndarray): Target values of shape (n_samples,).

        Returns:
            Self: The trained model.
        """

        # re-label the classes to real numbers 0, 1, etc.
        y_train, classes = map_class_labels(y_train)
        
        # convert the target vector into "one-hot" encoding
        y_train_one_hot = np.eye(len(classes))[y_train]

        # store original class labels
        self.classes = classes

        # get the number of samples and features:
        n, m = X_train.shape

        # define the shape of the network
        shape = (m, *self.hidden_shape, len(classes))

        # initialise the weights and biases
        weights = []
        biases = []
        for idx, curr_layer_size in enumerate(shape[1:], 1):

            # get the shape of the previous layer
            prev_layer_size = shape[idx-1]

            # randomise the current weights
            curr_weights = np.random.randn(prev_layer_size, curr_layer_size)
            weights.append(curr_weights)

            # set current biases to zero
            curr_biases = np.zeros(curr_layer_size)
            biases.append(curr_biases)
        
        # choose activation functions for each layer
        activations = [self.ReLU]*len(self.hidden_shape) + [self.softmax]

        # loop through the number of epochs
        for _ in range(self.epochs):

            # run a forward pass through the model, caching outputs from layers in the network
            y_pred, cached_outputs = self.forward_pass(X_train, weights, biases, activations)

            # calculate the gradient of the weighted input for the final layer
            delta_l = y_pred - y_train_one_hot

            # get the gradient of the weights and biases for the final layer
            grad_w = (1/n) * cached_outputs[-1].T @ delta_l
            grad_b = (1/n) * np.sum(delta_l, axis=0)
            
            # update the weights and biases for the final layer by performing the gradient descent step
            weights[-1] -= self.learning_rate * grad_w
            biases[-1] -= self.learning_rate * grad_b

            # loop through each of the hidden layers
            for i in range(len(weights)-2, -1, -1):
        
                # calculate the gradient of the weighted input for the current layer
                delta_l = (delta_l @ weights[i+1].T) * (cached_outputs[i+1] > 0).astype(int)

                # compute the gradients
                grad_w = (1/n) * cached_outputs[i].T @ delta_l
                grad_b = (1/n) * np.sum(delta_l, axis=0)

                # update the weights and biases for the current layers by performing the gradient descent step
                weights[i] -= self.learning_rate * grad_w
                biases[i] -= self.learning_rate * grad_b

        # save the fitted weights and biases
        self.weights = weights
        self.biases = biases
        self.activations = activations

        return self

    # run a forward pass (without caching the outputs) of the model
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each test sample.

        Parameters:
            X_test (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """

        # run the input through the neural network, at each layer multiplying by the weights, adding the biases and then applying the activation function
        z = X_test
        for w, b, a in zip(self.weights, self.biases, self.activations):
            z = a(z @ w + b)
        
        pred_probs = z
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
    
    def forward_pass(self, X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray], activations: list[Callable]) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Perform a forward pass through the network, caching the output of each layer.

        Args:
            X (np.ndarray): The data to pass through the network of shape (n_samples, n_features).
            weights (list[np.ndarray]): A list of neuron weights of each layer.
            biases (list[np.ndarray]): A list of neuron biases of each layer.
            activations (list[function]): The activation functions of each layer.

        Returns:
            np.ndarray: A tuple of the predicted output probabilities for the final layer and is list of cached outputs at every other layer.
        """

        cached_outputs = []
        z = X
        for w, b, a in zip(weights, biases, activations):
            cached_outputs.append(z)
            z = a(z @ w + b)
        return z, cached_outputs


    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the softmax activation function

        Args:
            z (np.ndarray): Input vector.

        Returns:
            np.ndarray: The result of applying the softmax
        """

        exp_z = np.exp(z)

        return exp_z/np.sum(exp_z, axis=1)[:, None]
    
    def ReLU(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function

        Args:
            z (np.ndarray): Input vector.

        Returns:
            np.ndarray: The result of applying the ReLU.
        """

        return np.max(np.array([np.zeros(z.shape), z]), axis=0)