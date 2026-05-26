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

        Parameters:
        - units: integer, number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()

        # Dense layer to be applied to the previous decoder hidden state
        self.W = tf.keras.layers.Dense(units=units)

        # Dense layer to be applied to the encoder hidden states
        self.U = tf.keras.layers.Dense(units=units)

        # Dense layer to be applied to the tanh of the sum of W and U
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Calculates the attention context vector and alignment weights.

        Parameters:
        - s_prev: tensor of shape (batch, units) containing the previous
                  decoder hidden state
        - hidden_states: tensor of shape (batch, input_seq_len, units)
                         containing the outputs of the encoder

        Returns:
        - context: tensor of shape (batch, units) containing context vector
        - weights: tensor of shape (batch, input_seq_len, 1) containing
                   attention weights
        """
        # Expand s_prev dimensions to (batch, 1, units) to broadcast
        # across the input_seq_len axis of hidden_states
        s_prev_time = tf.expand_dims(s_prev, axis=1)

        # Calculate alignment scores: score = V(tanh(W(s_prev) + U(h_i)))
        # Shape of score: (batch, input_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_time) + self.U(hidden_states)))

        # Calculate attention weights via softmax across the sequence length
        # Shape of weights: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector as the weighted sum of encoder hidden states
        # weights: (batch, input_seq_len, 1)
        # hidden_states: (batch, input_seq_len, units)
        context = weights * hidden_states

        # Sum along the input_seq_len axis to get final context vector
        # Shape of context: (batch, units)
        context = tf.reduce_sum(context, axis=1)

        return context, weights
