#!/usr/bin/env python3
"""
Module to create attention masks for a Transformer network.
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all padding and look-ahead masks for training/validation.

    Parameters:
    - inputs: tf.Tensor of shape (batch_size, seq_len_in) containing
              the input sentence tokens
    - target: tf.Tensor of shape (batch_size, seq_len_out) containing
              the target sentence tokens

    Returns:
    - encoder_mask: padding mask for the encoder
    - combined_mask: padding and look-ahead mask for decoder block 1
    - decoder_mask: padding mask for decoder block 2
    """
    # 1. Encoder Padding Mask
    # Finds where input values are equal to 0 (the pad token)
    enc_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    # Reshape to (batch_size, 1, 1, seq_len_in) for broadcasting
    encoder_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    # 2. Decoder Padding Mask (used in Cross-Attention block 2)
    # Reshape to (batch_size, 1, 1, seq_len_in)
    decoder_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    # 3. Combined Mask (used in Self-Attention block 1 of the Decoder)
    # Look-ahead mask to mask out future tokens in the target sequence
    size = tf.shape(target)[1]
    # Creates a lower triangular matrix of ones: shape (size, size)
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((size, size)), -1, 0
    )

    # Padding mask for the target sequence
    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    # Reshape to (batch_size, 1, 1, seq_len_out)
    dec_target_mask = dec_target_mask[:, tf.newaxis, tf.newaxis, :]

    # Take the element-wise maximum between look-ahead and target padding
    combined_mask = tf.maximum(look_ahead_mask, dec_target_mask)

    return encoder_mask, combined_mask, decoder_mask
