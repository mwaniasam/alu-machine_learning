#!/usr/bin/env python3
"""
Module defining an RNN Encoder for machine translation.
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    An RNN Encoder class that inherits from tensorflow.keras.layers.Layer.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor to initialize the encoder layer.

        Parameters:
        - vocab: integer, size of the input vocabulary
        - embedding: integer, dimensionality of the embedding vector
        - units: integer, number of hidden units in the RNN cell
        - batch: integer, the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        # Define Keras layers as public instance attributes
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

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros.

        Returns:
        - A tensor of shape (batch, units) containing initialized hidden states
        """
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        Processes the input sequence through embedding and GRU operations.

        Parameters:
        - x: tensor of shape (batch, input_seq_len) with input word indices
        - initial: tensor of shape (batch, units) with initial hidden state

        Returns:
        - outputs: tensor of shape (batch, input_seq_len, units)
        - hidden: tensor of shape (batch, units) containing last hidden state
        """
        # Pass input through the embedding layer
        embedded_x = self.embedding(x)

        # GRU layer with return_state=True outputs: (whole_seq, last_state)
        outputs, hidden = self.gru(embedded_x, initial_state=initial)

        return outputs, hidden
