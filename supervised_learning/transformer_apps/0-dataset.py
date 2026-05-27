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
        # Load the train and validation splits as supervised tuples
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

        # Create tokenizers from the training dataset split
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
        # Define maximum vocabulary size constraint
        vocab_size = 2 ** 15

        # Build Portuguese tokenizer from generator mapping
        tokenizer_pt = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=vocab_size
            )
        )

        # Build English tokenizer from generator mapping
        tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=vocab_size
            )
        )

        return tokenizer_pt, tokenizer_en
