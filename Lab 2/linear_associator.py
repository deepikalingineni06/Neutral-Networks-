

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        # Input dimensions
        self.D = input_dimensions
        # N dimensions
        self.N = number_of_nodes
        # Transfer function
        self.tf = transfer_function

        # Initilize weights
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed:
            np.random.seed(seed)

        self.weights = np.random.randn(self.N, self.D)

    def set_weights(self, W):
        """
        This function sets the weight matrix.
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        if W.shape == self.weights.shape:
            self.weights = W
        else:
            return -1

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        # Dot product
        pred = self.weights.dot(X)
        if self.tf == 'Hard_limit':
            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            pred = pred.astype(int)
        return pred

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        # Fit pseudo inverse weights
        self.weights = y.dot(np.linalg.pinv(X))

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        batch_x = np.split(X, batch_size, axis=1)
        batch_y = np.split(y, batch_size, axis=1)

        for epoch in range(num_epochs):
            for i in range(len(batch_x)):
                y_hat = self.predict(batch_x[i])
                mse = self.calculate_mean_squared_error(y_hat, batch_y[i])
                if learning == 'Delta':  # Default
                    self.weights += alpha * \
                        (np.subtract(batch_y[i], y_hat).dot(batch_x[i].T))
                elif learning == 'Filtered':  # In case the given rule is `Filtered`
                    self.weights = (1 - gamma) * self.weights + \
                        alpha * batch_y[i].dot(batch_x[i].T)
                elif learning == 'Unsupervised_hebb':  # In case the given rule is `Unsupervised_hebb`
                    self.weights += alpha * (y_hat.dot(batch_x[i].T))

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        # Predictions
        pred = self.predict(X)
        # Save mean squared error
        mean_squared_error = np.square(pred - y).mean()
        return mean_squared_error
