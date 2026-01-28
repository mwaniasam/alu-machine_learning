#!/usr/bin/env python3
"""Module for Neuron class"""
import numpy as np


class Neuron:
    """Neuron class for binary classification"""

    def __init__(self, nx):
        """
        Initialize a Neuron instance

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The activated output (self.__A)
        """
        # Calculate linear combination: z = W·X + b
        z = np.matmul(self.__W, X) + self.__b
        
        # Apply sigmoid activation function: A = 1 / (1 + e^(-z))
        self.__A = 1 / (1 + np.exp(-z))
        
        return self.__A

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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
            The neuron's prediction and the cost of the network
        """
        # Get the activated output using forward propagation
        A = self.forward_prop(X)
        
        # Convert activations to predictions (1 if >= 0.5, 0 otherwise)
        prediction = np.where(A >= 0.5, 1, 0)
        
        # Calculate the cost
        cost = self.cost(Y, A)
        
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output
            alpha: the learning rate

        Updates the private attributes __W and __b
        """
        # Number of examples
        m = X.shape[1]
        
        # Calculate dZ (derivative of cost with respect to Z)
        dZ = A - Y
        
        # Calculate gradients
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        
        # Update weights and bias
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
