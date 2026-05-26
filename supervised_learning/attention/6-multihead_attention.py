#!/usr/bin/env python3
"""
Module defining a Multi-Head Attention layer using TensorFlow.
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Performs Multi-Head Attention on academic transformer structures.
    """

    def __init__(self, dm, h):
        """
        Class constructor.

        Parameters:
        - dm: integer, dimensionality of the model
        - h: integer, number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Initialize all dense projections with glorot_uniform
        self.Wq = tf.keras.layers.Dense(
            units=dm,
            kernel_initializer='glorot_uniform'
        )
        self.Wk = tf.keras.layers.Dense(
            units=dm,
            kernel_initializer='glorot_uniform'
        )
        self.Wv = tf.keras.layers.Dense(
            units=dm,
            kernel_initializer='glorot_uniform'
        )
        self.linear = tf.keras.layers.Dense(
            units=dm,
            kernel_initializer='glorot_uniform'
        )

    def split_heads(self, x, batch_size):
        """
        Helper method to split the last dimension into (h, depth)
        and transpose the result to shape (batch, h, seq_len, depth).
        """
        # Reshape from (batch, seq_len, dm) to (batch, seq_len, h, depth)
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        # Transpose to (batch, h, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Executes multi-head attention processing logic.

        Parameters:
        - Q: tensor containing input for queries, shape (batch, seq_len_q, dk)
        - K: tensor containing input for keys, shape (batch, seq_len_v, dk)
        - V: tensor containing input for values, shape (batch, seq_len_v, dv)
        - mask: always None as per instructions

        Returns:
        - output: tensor of shape (batch, seq_len_q, dm)
        - weights: tensor of shape (batch, h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # 1. Project inputs through dense layers to get Q, K, V matrices
        q = self.Wq(Q)  # (batch, seq_len_q, dm)
        k = self.Wk(K)  # (batch, seq_len_v, dm)
        v = self.Wv(V)  # (batch, seq_len_v, dm)

        # 2. Split matrices into multiple heads
        q = self.split_heads(q, batch_size)  # (batch, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  # (batch, h, seq_len_v, depth)

        # 3. Calculate Scaled Dot-Product Attention across split heads
        # scaled_attention shape: (batch, h, seq_len_q, depth)
        # weights shape: (batch, h, seq_len_q, seq_len_v)
        scaled_attention, weights = sdp_attention(q, k, v, mask)

        # 4. Transpose heads back: (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 5. Concatenate heads back to original shape: (batch, seq_len_q, dm)
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm)
        )

        # 6. Apply final linear layer projection
        output = self.linear(concat_attention)

        return output, weights
