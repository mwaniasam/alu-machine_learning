#!/usr/bin/env python3
"""
Module defining a complete Transformer Encoder using TensorFlow.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    A full Transformer Encoder containing N structural encoder blocks.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.

        Parameters:
        - N: integer, the number of blocks in the encoder
        - dm: integer, the dimensionality of the model
        - h: integer, the number of heads
        - hidden: integer, number of hidden units in the dense layer
        - input_vocab: integer, the size of the input vocabulary
        - max_seq_len: integer, the maximum sequence length possible
        - drop_rate: float, the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm

        # Input Embedding layer mapping vocabulary space to model depth
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab,
            output_dim=dm
        )

        # Pre-calculated numpy array of positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List containing N sequential encoder block layers
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

        # Dropout layer applied immediately after adding positional encodings
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
        Processes input sequence indices through token embedding,
        positional alignment, dropout, and consecutive encoder blocks.

        Parameters:
        - x: tensor of shape (batch, input_seq_len) containing input indices
        - training: boolean to determine if the model is training
        - mask: the mask to be applied for multi-head attention

        Returns:
        - A tensor of shape (batch, input_seq_len, dm) with encoder outputs
        """
        seq_len = tf.shape(x)[1]

        # 1. Generate dense word embeddings from input indices
        # Output shape: (batch, seq_len, dm)
        x = self.embedding(x)

        # Scale embeddings by square root of model dimension
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Add structural positional encodings up to current runtime length
        # Sliced array shape: (seq_len, dm)
        x += self.positional_encoding[:seq_len, :]

        # 3. Apply initial dropout regularization
        x = self.dropout(x, training=training)

        # 4. Sequentially process data through each of the N Encoder Blocks
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
