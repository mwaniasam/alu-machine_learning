#!/usr/bin/env python3
"""
Module defining a complete Transformer Decoder using TensorFlow.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    A full Transformer Decoder containing N structural decoder blocks.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.

        Parameters:
        - N: integer, the number of blocks in the decoder
        - dm: integer, the dimensionality of the model
        - h: integer, the number of heads
        - hidden: integer, number of hidden units in dense layer
        - target_vocab: integer, the size of the target vocabulary
        - max_seq_len: integer, the maximum sequence length possible
        - drop_rate: float, the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm

        # Target Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab,
            output_dim=dm
        )

        # Pre-calculated numpy array of positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List containing N sequential decoder block layers
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

        # Dropout layer applied immediately after adding position data
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Processes target sequence indices through token embedding,
        positional alignment, dropout, and consecutive decoder blocks.

        Parameters:
        - x: tensor of shape (batch, target_seq_len) target indices
        - encoder_output: tensor, output from the encoder
        - training: boolean to determine if the model is training
        - look_ahead_mask: mask applied to the masked self-attention
        - padding_mask: mask applied to the cross-attention layer

        Returns:
        - A tensor of shape (batch, target_seq_len, dm) decoder output
        """
        target_seq_len = tf.shape(x)[1]

        # 1. Generate dense embeddings from target token indices
        x = self.embedding(x)

        # Scale embeddings by the square root of model dimension
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 2. Add structural positional encodings up to target sequence length
        x += self.positional_encoding[:target_seq_len, :]

        # 3. Apply initial dropout regularization
        x = self.dropout(x, training=training)

        # 4. Sequentially process data through each of the N Decoder Blocks
        for i in range(self.N):
            x = self.blocks[i](
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x
