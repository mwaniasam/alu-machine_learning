#!/usr/bin/env python3
"""
Module containing the tf_idf function matching TfidfVectorizer specs.
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix using smooth IDF and L2 normalization.

    Parameters:
    sentences (list): A list of sentences to analyze.
    vocab (list): A list of vocabulary words to use for the analysis.
                  If None, all words within sentences are used.

    Returns:
    embeddings (numpy.ndarray): Matrix of shape (s, f) with TF-IDF values.
    features (list): Sorted list of the features used for the embeddings.
    """
    cleaned_sentences = []

    # Clean text exactly identical to Task 0
    for sentence in sentences:
        modified_text = sentence.lower().replace("'s", "")
        for char in ".,!?::;\n\"":
            modified_text = modified_text.replace(char, " ")
        cleaned_sentences.append(modified_text.split())

    # Build or extract feature vocabulary
    if vocab is None:
        unique_words = set()
        for words in cleaned_sentences:
            unique_words.update(words)
        features = sorted(list(unique_words))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)

    # Step 1: Compute raw Term Frequency (TF) matrix
    tf = np.zeros((s, f))
    feature_indices = {word: i for i, word in enumerate(features)}

    for row_idx, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_indices:
                col_idx = feature_indices[word]
                tf[row_idx, col_idx] += 1

    # Step 2: Compute Document Frequency (DF)
    df = np.sum(tf > 0, axis=0)

    # Step 3: Compute smooth IDF matching scikit-learn equation:
    # idf = ln((1 + s) / (1 + df)) + 1
    idf = np.log((1 + s) / (1 + df)) + 1

    # Step 4: Multiply TF by IDF
    embeddings = tf * idf

    # Step 5: Apply L2 row-normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Safe division avoiding zeros
    embeddings = np.divide(embeddings, norms, out=embeddings,
                           where=norms > 0)

    return embeddings, features
