#!/usr/bin/env python3
"""
Module defining a Transformer Encoder Block using TensorFlow.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    An encoder block for a transformer network.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor.

        Parameters:
        - dm: integer, the dimensionality of the model
        - h: integer, the number of heads
        - hidden: integer, number of hidden units in the fully connected layer
        - drop_rate: float, the dropout rate
        """
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(dm, h)

        # Feed-forward hidden dense layer with ReLU activation
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation='relu',
            kernel_initializer='glorot_uniform'
        )

        # Feed-forward output dense layer
        self.dense_output = tf.keras.layers.Dense(
            units=dm,
            kernel_initializer='glorot_uniform'
        )

        # Layer Normalization layers with exact epsilon requirement
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Processes input through multi-head attention and feed-forward networks
        with residual connections and layer normalization.

        Parameters:
        - x: tensor of shape (batch, input_seq_len, dm) containing the input
        - training: boolean to determine if the model is training
        - mask: the mask to be applied for multi-head attention

        Returns:
        - A tensor of shape (batch, input_seq_len, dm) containing the output
        """
        # 1. First Sub-Layer: Multi-Head Attention + Dropout + Residual Connection
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 2. Second Sub-Layer: Position-wise Feed-Forward Network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual Connection + Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
