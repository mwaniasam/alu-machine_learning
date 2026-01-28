#!/usr/bin/env python3
"""Calculates the loss of a prediction"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss

    Arguments:
        y (tf.Tensor): placeholder for true labels (one-hot)
        y_pred (tf.Tensor): tensor with predicted logits

    Returns:
        tf.Tensor: scalar tensor containing the loss
    """
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )

    return loss
