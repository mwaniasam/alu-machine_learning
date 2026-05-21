#!/usr/bin/env python3
"""
Module containing the bag_of_words function.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    sentences (list): A list of sentences to analyze.
    vocab (list): A list of vocabulary words to use for the analysis.
                  If None, all words within sentences are used.

    Returns:
    embeddings (numpy.ndarray): Matrix of shape (s, f) with word counts.
    features (list): Sorted list of the features used for the embeddings.
    """
    cleaned_sentences = []

    # Step 1: Preprocess sentences - lowercase and handle trailing punctuation
    for sentence in sentences:
        words = []
        for word in sentence.lower().split():
            # Strip trailing punctuation marks but keep internal apostrophes
            while word and word[-1] in ".!?,;:":
                word = word[:-1]
            if word:
                words.append(word)
        cleaned_sentences.append(words)

    # Step 2: Build or extract feature vocabulary
    if vocab is None:
        unique_words = set()
        for words in cleaned_sentences:
            unique_words.update(words)
        features = sorted(list(unique_words))
    else:
        features = vocab

    # Step 3: Populate the occurrence matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Create a word-to-index mapping for fast lookups
    feature_indices = {word: i for i, word in enumerate(features)}

    for row_idx, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_indices:
                col_idx = feature_indices[word]
                embeddings[row_idx, col_idx] += 1

    return embeddings, features
