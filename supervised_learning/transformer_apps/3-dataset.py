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

    def __init__(self, batch_size, max_len):
        """
        Initializes datasets, creates tokenizers, and sets up pipelines.

        Parameters:
        - batch_size: integer, batch size for training/validation
        - max_len: integer, maximum number of tokens allowed per sentence
        """
        # Load the raw splits alongside their underlying registration metadata
        raw_train, info = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True,
            with_info=True
        )
        raw_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        t_pt, t_en = self.tokenize_dataset(raw_train)
        self.tokenizer_pt = t_pt
        self.tokenizer_en = t_en

        # Map tokenization across datasets before optimization steps
        data_train = raw_train.map(self.tf_encode)
        data_valid = raw_valid.map(self.tf_encode)

        # Build training pipeline
        # 1. Filter out pairs exceeding maximum length boundaries
        data_train = data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.shape(pt)[0] <= max_len,
                tf.shape(en)[0] <= max_len
            )
        )
        # 2. Cache dataset for execution performance speedups
        data_train = data_train.cache()
        # 3. Shuffle across the entire dataset size allocation space
        buffer_size = info.splits['train'].num_examples
        data_train = data_train.shuffle(buffer_size)
        # 4. Group sequence batches together with explicit 0 padding
        data_train = data_train.padded_batch(batch_size)
        # 5. Pipeline prefetching via backend autotuning optimization
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Build validation pipeline
        data_valid = data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.shape(pt)[0] <= max_len,
                tf.shape(en)[0] <= max_len
            )
        )
        self.data_valid = data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.
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
        """
        pt_encoded = self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        en_encoded = self.tokenizer_en.encode(en.numpy().decode('utf-8'))

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

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the python-based encode instance method.
        """
        pt_tensor, en_tensor = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tensor.set_shape([None])
        en_tensor.set_shape([None])

        return pt_tensor, en_tensor
