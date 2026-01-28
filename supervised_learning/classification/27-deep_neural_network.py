#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep neural network for multiclass classification"""

    def __init__(self, nx, layers):
        """
        Initialize a DeepNeuralNetwork instance

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if layers is not a list or empty
            TypeError: if elements in layers are not positive integers
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Validate all elements in layers are positive integers
        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

        # Set private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases for each layer
        layer_sizes = [nx] + layers
        for l in range(1, self.__L + 1):
            # He et al. initialization for weights
            self.__weights['W' + str(l)] = (
                np.random.randn(layer_sizes[l], layer_sizes[l - 1]) *
                np.sqrt(2 / layer_sizes[l - 1])
            )
            # Initialize biases to zeros
            self.__weights['b' + str(l)] = np.zeros((layer_sizes[l], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples

        Returns:
            The output of the neural network and the cache
        """
        # Store input in cache as A0
        self.__cache['A0'] = X

        # Forward propagation through all layers
        A = X
        for l in range(1, self.__L + 1):
            # Get weights and bias for current layer
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]

            # Calculate linear combination: Z = W * A_prev + b
            Z = np.matmul(W, A) + b

            # For output layer, use softmax; for hidden layers, use sigmoid
            if l == self.__L:
                # Softmax activation for output layer
                # Subtract max for numerical stability
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Sigmoid activation for hidden layers
                A = 1 / (1 + np.exp(-Z))

            # Store activated output in cache
            self.__cache['A' + str(l)] = A

        # Return final output and cache
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using cross-entropy

        Args:
            Y: numpy.ndarray with shape (classes, m) containing correct
               labels (one-hot encoded)
            A: numpy.ndarray with shape (classes, m) containing
               activated output

        Returns:
            The cost
        """
        # Number of examples
        m = Y.shape[1]

        # Cross-entropy cost function for multiclass classification
        # Cost = -1/m * Σ Σ Y_ij * log(A_ij)
        # Add small epsilon to avoid log(0)
        cost = -1 / m * np.sum(Y * np.log(A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples
            Y: numpy.ndarray with shape (classes, m) containing correct labels
               (one-hot encoded)

        Returns:
            The neural network's prediction (one-hot encoded) and the cost
        """
        # Get the activated output using forward propagation
        A, _ = self.forward_prop(X)

        # Convert activations to one-hot predictions
        # Find the index of the maximum value for each example
        max_indices = np.argmax(A, axis=0)

        # Create one-hot encoded prediction
        prediction = np.zeros_like(A)
        prediction[max_indices, np.arange(A.shape[1])] = 1

        # Calculate the cost
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (classes, m) containing correct labels
               (one-hot encoded)
            cache: dictionary containing all intermediary values
            alpha: the learning rate

        Updates the private attribute __weights
        """
        # Number of examples
        m = Y.shape[1]

        # Start backpropagation from the output layer
        # dZ for output layer with softmax: dZ_L = A_L - Y
        dZ = cache['A' + str(self.__L)] - Y

        # Backpropagate through all layers
        for l in range(self.__L, 0, -1):
            # Get activation from previous layer
            A_prev = cache['A' + str(l - 1)]

            # Calculate gradients
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Update weights and biases
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

            # Calculate dZ for previous layer (if not the input layer)
            if l > 1:
                W = self.__weights['W' + str(l)]
                A_prev = cache['A' + str(l - 1)]
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples
            Y: numpy.ndarray with shape (classes, m) containing correct labels
               (one-hot encoded)
            iterations: the number of iterations to train over
            alpha: the learning rate
            verbose: boolean to print training information
            graph: boolean to plot training cost
            step: iterations between printing/plotting

        Returns:
            The evaluation of the training data after iterations

        Raises:
            TypeError: if iterations is not an integer
            ValueError: if iterations is not positive
            TypeError: if alpha is not a float
            ValueError: if alpha is not positive
            TypeError: if step is not an integer
            ValueError: if step is not positive or > iterations
        """
        # Validate iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Validate step only if verbose or graph is True
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Lists to store costs and iterations for plotting
        costs = []
        iteration_list = []

        # Training loop
        for i in range(iterations + 1):
            # Forward propagation
            A, cache = self.forward_prop(X)

            # Print and/or record cost at step intervals and at 0 and last
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, A)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

                if graph:
                    costs.append(cost)
                    iteration_list.append(i)

            # Perform gradient descent (except after last iteration)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # Plot the graph if requested
        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation after training
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename: the file to which the object should be saved
                     If filename does not have .pkl extension, it will be added
        """
        # Add .pkl extension if not present
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'

        # Save the object using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename: the file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
