#!/usr/bin/env python3
"""Module for NeuralNetwork class"""
import numpy as np


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification"""

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance

        Args:
            nx: number of input features
            nodes: number of nodes in the hidden layer

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
            TypeError: if nodes is not an integer
            ValueError: if nodes is less than 1
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        
        # Initialize private weights and biases for hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        
        # Initialize private weights and biases for output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights vector of hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias of hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output of hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights vector of output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output of output neuron (prediction)"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples

        Returns:
            The private attributes __A1 and __A2
        """
        # Calculate hidden layer
        # Z1 = W1 · X + b1
        Z1 = np.matmul(self.__W1, X) + self.__b1
        
        # Apply sigmoid activation: A1 = 1 / (1 + e^(-Z1))
        self.__A1 = 1 / (1 + np.exp(-Z1))
        
        # Calculate output layer
        # Z2 = W2 · A1 + b2
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        
        # Apply sigmoid activation: A2 = 1 / (1 + e^(-Z2))
        self.__A2 = 1 / (1 + np.exp(-Z2))
        
        return self.__A1, self.__A2
