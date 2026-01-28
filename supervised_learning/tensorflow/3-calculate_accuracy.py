#!/usr/bin/env python3
"""Calculates the accuracy of a prediction"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of predictions

    Arguments:
        y (tf.Tensor): placeholder for true labels (one-hot)
        y_pred (tf.Tensor): tensor with predicted labels

    Returns:
        tf.Tensor: scalar tensor containing accuracy
    """
    correct_predictions = tf.equal(
        tf.argmax(y, axis=1),
        tf.argmax(y_pred, axis=1)
    )

    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32)
    )

    return accuracy
