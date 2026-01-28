#!/usr/bin/env python3
"""Creates the training operation for a neural network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation using gradient descent

    Arguments:
        loss (tf.Tensor): loss of the network
        alpha (float): learning rate

    Returns:
        tf.Operation: training operation
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)

    return train_op
