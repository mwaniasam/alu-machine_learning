#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

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

            # Apply sigmoid activation: A = 1 / (1 + e^(-Z))
            A = 1 / (1 + np.exp(-Z))

            # Store activated output in cache
            self.__cache['A' + str(l)] = A

        # Return final output and cache
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output

        Returns:
            The cost
        """
        # Number of examples
        m = Y.shape[1]

        # Logistic regression cost function
        # Cost = -1/m * Σ[Y*log(A) + (1-Y)*log(1-A)]
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

        return cost
