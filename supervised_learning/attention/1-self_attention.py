#!/usr/bin/env python3
"""
Module defining a Self Attention layer for machine translation.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Calculates Bahdanau attention for a sequence-to-sequence model.
    """

    def __init__(self, units):
        """
        Class constructor.
        """
        super(SelfAttention, self).__init__()

        # Order of instantiation matters for internal random weight seeding
        self.W = tf.keras.layers.Dense(
            units=units,
            kernel_initializer='glorot_uniform'
        )
        self.U = tf.keras.layers.Dense(
            units=units,
            kernel_initializer='glorot_uniform'
        )
        self.V = tf.keras.layers.Dense(
            units=1,
            kernel_initializer='glorot_uniform'
        )

    def call(self, s_prev, hidden_states):
        """
        Calculates the attention context vector and alignment weights.
        """
        s_prev_time = tf.expand_dims(s_prev, axis=1)

        score = self.V(tf.nn.tanh(self.W(s_prev_time) + self.U(hidden_states)))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
