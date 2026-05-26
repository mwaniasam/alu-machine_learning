#!/usr/bin/env python3
"""
Module defining a Transformer Decoder Block using TensorFlow.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    A decoder block for a transformer network.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor.

        Parameters:
        - dm: integer, the dimensionality of the model
        - h: integer, the number of heads
        - hidden: integer, number of hidden units in the dense layer
        - drop_rate: float, the dropout rate
        """
        super(DecoderBlock, self).__init__()

        # First Multi-Head Attention layer (Masked Self-Attention)
        self.mha1 = MultiHeadAttention(dm, h)

        # Second Multi-Head Attention layer (Encoder-Decoder Attention)
        self.mha2 = MultiHeadAttention(dm, h)

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

        # Layer Normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Processes input tensors through targeted sub-layers, including
        masked self-attention, cross-attention, and position-wise FFN.
        """
        # 1. First Sub-Layer: Masked Self-Attention
        # Queries, Keys, and Values all come from the decoder input x
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Second Sub-Layer: Encoder-Decoder Cross-Attention
        # Queries come from out1; Keys and Values come from encoder_output
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 3. Third Sub-Layer: Position-wise Feed-Forward Network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
