#!/usr/bin/env python3
"""Creates placeholders for a neural network"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for input data and labels

    Arguments:
        nx (int): number of feature columns
        classes (int): number of classes

    Returns:
        x (tf.placeholder): placeholder for input data
        y (tf.placeholder): placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
