#!/usr/bin/env python3
"""
Module containing the fasttext_model training function.
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim FastText model.

    Parameters:
    sentences (list): A list of sentences to be trained on.
    size (int): The dimensionality of the embedding layer.
    min_count (int): The minimum number of word occurrences for training.
    negative (int): The size of negative sampling.
    window (int): Maximum distance between current and predicted word.
    cbow (bool): Determines training type. True for CBOW, False for Skip-gram.
    iterations (int): The number of iterations (epochs) to train over.
    seed (int): Seed for the random number generator.
    workers (int): The number of worker threads to train the model.

    Returns:
    model: The trained Gensim FastText model instance.
    """
    # Map cbow flag to Gensim's internal structural choice (0: CBOW, 1: Skip)
    sg_flag = 0 if cbow else 1

    # Initialize and train the FastText subword model
    model = FastText(sentences=sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     seed=seed,
                     workers=workers,
                     negative=negative,
                     sg=sg_flag,
                     iter=iterations)

    return model
