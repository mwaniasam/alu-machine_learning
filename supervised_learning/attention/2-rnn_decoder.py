#!/usr/bin/env python3
"""
Module defining an RNN Decoder with Attention for machine translation.
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    An RNN Decoder class that inherits from tensorflow.keras.layers.Layer.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor to initialize the decoder layer.
        """
        super(RNNDecoder, self).__init__()

        # Maintain this exact layer instantiation sequence
        self.attention = SelfAttention(units)

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(
            units=vocab,
            kernel_initializer='glorot_uniform'
        )

    def call(self, x, s_prev, hidden_states):
        """
        Decodes a single time step using attention and a GRU cell.
        """
        context, _ = self.attention(s_prev, hidden_states)

        x_embed = self.embedding(x)

        context_expanded = tf.expand_dims(context, axis=1)

        combined_input = tf.concat([context_expanded, x_embed], axis=2)

        outputs, s = self.gru(combined_input, initial_state=s_prev)

        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        y = self.F(outputs)

        return y, s
