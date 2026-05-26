#!/usr/bin/env python3
"""
Module to calculate Scaled Dot Product Attention using TensorFlow.
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Parameters:
    - Q: tensor containing the query matrix, shape (..., seq_len_q, dk)
    - K: tensor containing the key matrix, shape (..., seq_len_v, dk)
    - V: tensor containing the value matrix, shape (..., seq_len_v, dv)
    - mask: tensor containing the optional mask, broadcastable to
            (..., seq_len_q, seq_len_v), defaults to None

    Returns:
    - output: tensor containing the scaled dot product attention,
              shape (..., seq_len_q, dv)
    - weights: tensor containing the attention weights,
               shape (..., seq_len_q, seq_len_v)
    """
    # Get the dimension of the keys dk (the last dimension of Q or K)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Multiply Q by transposed K
    # Q: (..., seq_len_q, dk) x K_T: (..., dk, seq_len_v)
    # Resulting shape: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot products by dividing by the square root of dk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # If mask is provided, multiply by -1e9 and add it to the scaled matrix
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax along the last axis to get probabilities (weights)
    # Shape: (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply weights by V
    # weights: (..., seq_len_q, seq_len_v) x V: (..., seq_len_v, dv)
    # Resulting output shape: (..., seq_len_q, dv)
    output = tf.matmul(weights, V)

    return output, weights
