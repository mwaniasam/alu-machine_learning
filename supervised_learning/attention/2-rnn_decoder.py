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

        Parameters:
        - vocab: integer, size of the output vocabulary
        - embedding: integer, dimensionality of the embedding vector
        - units: integer, number of hidden units in the RNN cell
        - batch: integer, the batch size
        """
        super(RNNDecoder, self).__init__()

        # Initialize the attention layer
        self.attention = SelfAttention(units)

        # Keras Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        # Keras GRU layer
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        # Keras Fully Connected (Dense) layer to output vocabulary scores
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Decodes a single time step using attention and a GRU cell.

        Parameters:
        - x: tensor of shape (batch, 1) containing the previous word index
        - s_prev: tensor of shape (batch, units) containing previous hidden state
        - hidden_states: tensor of shape (batch, input_seq_len, units)
                         containing the outputs of the encoder

        Returns:
        - y: tensor of shape (batch, vocab) containing the output word scores
        - s: tensor of shape (batch, units) containing the new hidden state
        """
        # 1. Calculate the context vector using the attention mechanism
        # context shape: (batch, units)
        context, _ = self.attention(s_prev, hidden_states)

        # 2. Pass the input token through the embedding layer
        # x shape: (batch, 1) -> x_embed shape: (batch, 1, embedding)
        x_embed = self.embedding(x)

        # 3. Concatenate the context vector with x in that order
        # Expand context from (batch, units) to (batch, 1, units)
        context_expanded = tf.expand_dims(context, axis=1)

        # Concatenate along the features axis (axis=2)
        # combined shape: (batch, 1, units + embedding)
        combined_input = tf.concat([context_expanded, x_embed], axis=2)

        # 4. Pass the concatenated vector into the GRU cell
        # outputs shape: (batch, 1, units), s shape: (batch, units)
        outputs, s = self.gru(combined_input, initial_state=s_prev)

        # 5. Reshape outputs to pass through the final Dense layer
        # outputs shape from (batch, 1, units) to (batch, units)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        # y shape: (batch, vocab)
        y = self.F(outputs)

        return y, s
