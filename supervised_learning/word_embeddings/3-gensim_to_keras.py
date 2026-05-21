#!/usr/bin/env python3
"""
Module containing the gensim_to_keras conversion function.
"""


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model into a trainable Keras
    Embedding layer.

    Parameters:
    model: A trained gensim word2vec model instance.

    Returns:
    keras.layers.embeddings.Embedding: An initialized, trainable Keras
                                       Embedding layer holding the weights
                                       from the Word2Vec model.
    """
    return model.wv.get_keras_embedding(True)
