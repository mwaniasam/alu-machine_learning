#!/usr/bin/env python3
"""
Module defining a complete Transformer Network using TensorFlow.
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    A full Transformer Network combining an Encoder and a Decoder.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor.

        Parameters:
        - N: integer, number of blocks in encoder and decoder
        - dm: integer, the dimensionality of the model
        - h: integer, the number of heads
        - hidden: integer, number of hidden units in dense layers
        - input_vocab: integer, the size of the input vocabulary
        - target_vocab: integer, the size of the target vocabulary
        - max_seq_input: integer, maximum sequence length for input
        - max_seq_target: integer, maximum sequence length for target
        - drop_rate: float, the dropout rate
        """
        super(Transformer, self).__init__()

        # Full Encoder component
        self.encoder = Encoder(
            N=N,
            dm=dm,
            h=h,
            hidden=hidden,
            input_vocab=input_vocab,
            max_seq_len=max_seq_input,
            drop_rate=drop_rate
        )

        # Full Decoder component
        self.decoder = Decoder(
            N=N,
            dm=dm,
            h=h,
            hidden=hidden,
            target_vocab=target_vocab,
            max_seq_len=max_seq_target,
            drop_rate=drop_rate
        )

        # Final linear mapping layer to output vocabulary logit spaces
        self.linear = tf.keras.layers.Dense(
            units=target_vocab,
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Executes a forward pass through the complete Transformer pipeline.

        Parameters:
        - inputs: tensor of shape (batch, input_seq_len) input tokens
        - target: tensor of shape (batch, target_seq_len) target tokens
        - training: boolean to determine if model is training
        - encoder_mask: padding mask applied to the encoder self-attention
        - look_ahead_mask: mask applied to the decoder self-attention
        - decoder_mask: padding mask applied to decoder cross-attention

        Returns:
        - A tensor of shape (batch, target_seq_len, target_vocab) containing
          the logit projections for translation/generation targets
        """
        # 1. Forward pass through the full Encoder block sequence
        # enc_output shape: (batch, input_seq_len, dm)
        enc_output = self.encoder(inputs, training, encoder_mask)

        # 2. Forward pass through the full Decoder block sequence
        # dec_output shape: (batch, target_seq_len, dm)
        dec_output = self.decoder(
            target,
            enc_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        # 3. Project hidden outputs into complete target logit shapes
        # final_output shape: (batch, target_seq_len, target_vocab)
        final_output = self.linear(dec_output)

        return final_output
