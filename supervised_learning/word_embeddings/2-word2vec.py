#!/usr/bin/env python3
"""
Module containing the word2vec_model training function.
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim Word2Vec model.

    Parameters:
    sentences (list): A list of sentences to be trained on.
    size (int): The dimensionality of the embedding layer.
    min_count (int): The minimum number of word occurrences for training.
    window (int): Maximum distance between current and predicted word.
    negative (int): The size of negative sampling.
    cbow (bool): Determines training type. True for CBOW, False for Skip-gram.
    iterations (int): The number of iterations (epochs) to train over.
    seed (int): Seed for the random number generator.
    workers (int): The number of worker threads to train the model.

    Returns:
    model: The trained Gensim Word2Vec model instance.
    """
    # Map boolean cbow flag to Gensim's internal sg parameter (1=Skip-gram, 0=CBOW)
    sg_flag = 0 if cbow else 1

    # Instantiate and train the Word2Vec model
    # Gensim 3.8.x uses size, window, min_count, seed, workers, negative, sg, iter
    model = Word2Vec(sentences=sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     seed=seed,
                     workers=workers,
                     negative=negative,
                     sg=sg_flag,
                     iter=iterations)

    return model
