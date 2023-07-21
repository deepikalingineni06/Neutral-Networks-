

# %tensorflow_version 2.x (2.8.0 used)
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
            Initialize multi-layer neural network
            :param input_dimension: The number of dimensions for each input data sample
        """
        self.D = input_dimension
        self.weights, self.B = list(), list()

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
            This function adds a dense layer to the neural network
            :param num_nodes: number of nodes in the layer
            :param transfer_function: Activation function for the layer.
            Possible values are:
                > 'Linear'
                > 'Relu'
                > 'Sigmoid'.
            :return: None
        """
        self.trans_func = transfer_function
        self.weights.append(tf.random.uniform(
            shape=(self.D, num_nodes), dtype='float32'))
        self.D = num_nodes
        self.B.append(tf.random.uniform(shape=(num_nodes,), dtype='float32'))

    def get_weights_without_biases(self, layer_number):
        """
            This function should return the weight matrix (without biases) for layer layer_number.
            layer numbers start from zero.
            :param layer_number: Layer number starting from layer 0. This means that the first layer with
            activation function is layer zero
            :return: Weight matrix for the given layer (not including the biases).
            Note that the shape of the weight matrix should be
            [input_dimensions][number of nodes]
        """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
            This function should return the biases for layer layer_number.
            layer numbers start from zero.
            This means that the first layer with activation function is layer zero
            :param layer_number: Layer number starting from layer 0
            :return: Weight matrix for the given layer (not including the biases).
            Note that the biases shape should be [1][number_of_nodes]
        """
        return self.B[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
            This function sets the weight matrix for layer layer_number.
            layer numbers start from zero.
            This means that the first layer with activation function is layer zero
            :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
            [input_dimensions][number of nodes]
            :param layer_number: Layer number starting from layer 0
            :return: none
        """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
            This function sets the biases for layer layer_number.
            layer numbers start from zero.
            This means that the first layer with activation function is layer zero
            :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
            :param layer_number: Layer number starting from layer 0
            :return: none
        """
        self.B[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
            This function calculates the sparse softmax cross entropy loss.
            :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
            the desired (true) class.
            :param y_hat: Array of actual output values [n_samples][number_of_classes].
            :return: loss
        """
        sparse_ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=y_hat)
        return tf.reduce_mean(sparse_ce_loss)

    def predict(self, X):
        """
            Given array of inputs, this function calculates the output of the multi-layer network.
            :param X: Array of input [n_samples,input_dimensions].
            :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        f = {
            "Linear": lambda x: x,
            "Relu": lambda x: tf.nn.relu(x),
            "Sigmoid": lambda x: tf.nn.sigmoid(x),
        }
        for i in range(len(self.weights)):
            mX = tf.matmul(X, self.weights[i])  # slope * X
            y = tf.add(mX, self.B[i])  # output: y = mX + B
            X = f[self.trans_func](y)  # activation function on X
        return X

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
            Given a batch of data, and the necessary hyperparameters, this function trains
            the neural network by adjusting the weights and biases of all the layers.
            :param X: Array of input [n_samples,input_dimensions]
            :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
            the desired (true) class.
            :param batch_size: number of samples in a batch
            :param num_epochs: Number of times training should be repeated over all input data
            :param alpha: Learning rate
            :return: None
        """
        for _epoch in range(num_epochs):
            for i in range(0, X_train.shape[1], batch_size):
                offset = i + batch_size
                li = X_train.shape[1] if offset > X_train.shape[1] else offset

                batch_X = X_train[i:li]
                batch_y = y_train[i:li]

                pred = self.predict(batch_X)
                loss = self.calculate_loss(batch_y, pred)

                with tf.GradientTape() as gt:
                    W, B = gt.gradient(loss, [self.weights, self.B])

                for j in range(len(self.weights)):
                    if W[j] != None and B[j] != None:
                        self.weights[j].assign_sub(alpha*W[j])
                        self.B[j].assign_sub(alpha*B[j])
                    else:
                        self.weights[j].assign_sub(alpha*self.weights[j])
                        self.B[j].assign_sub(alpha*self.B[j])

    def calculate_percent_error(self, X, y):
        """
            Given input samples and corresponding desired (true) output as indexes,
            this method calculates the percent error.
            For each input sample, if the predicted class output is not the same as the desired class,
            then it is considered one error. Percent error is number_of_errors/ number_of_samples.
            Note that the predicted class is the index of the node with maximum output.
            :param X: Array of input [n_samples,input_dimensions]
            :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
            the desired (true) class.
            :return percent_error
        """
        pred = self.predict(X)
        expn = [np.argmax(i, axis=0) for i in pred]
        error = y - expn

        count = 0
        for i in range(len(error)):
            if error[i] != 0:
                count += (1 / len(error))
        return count

    def calculate_confusion_matrix(self, X, y):
        """
            Given input samples and corresponding desired (true) outputs as indexes,
            this method calculates the confusion matrix.
            :param X: Array of input [n_samples,input_dimensions]
            :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
            the desired (true) class.
            :return confusion_matrix[number_of_classes,number_of_classes].
            Confusion matrix should be shown as the number of times that
            an image of class n is classified as class m.
        """
        cnf_mtx = np.ones((self.D, self.D))
        for actl, predn in zip(y, [np.argmax(i, axis=0) for i in self.predict(X)]):
            if actl != None and predn != None:
                cnf_mtx[actl][predn] += 1
        return cnf_mtx
