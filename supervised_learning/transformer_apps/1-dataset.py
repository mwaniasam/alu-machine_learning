#!/usr/bin/env python3
"""
Module defining a Dataset class for machine translation pipeline setup.
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and prepares a dataset for Portuguese to English translation.
    """

    def __init__(self):
        """
        Initializes datasets and creates sub-word tokenizers.
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        t_pt, t_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = t_pt
        self.tokenizer_en = t_en

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Parameters:
        - data: tf.data.Dataset tuple format (pt, en) tensors

        Returns:
        - tokenizer_pt: Portuguese sub-word tokenizer
        - tokenizer_en: English sub-word tokenizer
        """
        vocab_size = 2 ** 15

        tokenizer_pt = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=vocab_size
            )
        )

        tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=vocab_size
            )
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into token IDs with start/end markers.

        Parameters:
        - pt: tf.Tensor containing the Portuguese sentence string
        - en: tf.Tensor containing the English sentence string

        Returns:
        - pt_tokens: list containing the marked Portuguese token IDs
        - en_tokens: list containing the marked English token IDs
        """
        # Extract underlying strings and encode to subword lists
        pt_encoded = self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        en_encoded = self.tokenizer_en.encode(en.numpy().decode('utf-8'))

        # Add start token (vocab_size) and end token (vocab_size + 1)
        pt_tokens = (
            [self.tokenizer_pt.vocab_size] +
            pt_encoded +
            [self.tokenizer_pt.vocab_size + 1]
        )

        en_tokens = (
            [self.tokenizer_en.vocab_size] +
            en_encoded +
            [self.tokenizer_en.vocab_size + 1]
        )

        return pt_tokens, en_tokens
