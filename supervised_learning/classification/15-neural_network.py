#!/usr/bin/env python3
"""Module for NeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt


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
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
            The neural network's prediction and the cost
        """
        # Get the activated outputs using forward propagation
        _, A2 = self.forward_prop(X)
        
        # Convert activations to predictions (1 if >= 0.5, 0 otherwise)
        prediction = np.where(A2 >= 0.5, 1, 0)
        
        # Calculate the cost
        cost = self.cost(Y, A2)
        
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A1: output of the hidden layer
            A2: predicted output
            alpha: the learning rate

        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        # Number of examples
        m = X.shape[1]
        
        # Backpropagation for output layer
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Backpropagation for hidden layer
        # dZ1 = W2.T * dZ2 * A1 * (1 - A1)  (sigmoid derivative)
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Update weights and biases
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx is the number of input features
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels
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
            A1, A2 = self.forward_prop(X)
            
            # Print and/or record cost at step intervals and at 0 and last
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, A2)
                
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                
                if graph:
                    costs.append(cost)
                    iteration_list.append(i)
            
            # Perform gradient descent (except after last iteration)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)
        
        # Plot the graph if requested
        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        # Return evaluation after training
        return self.evaluate(X, Y)
