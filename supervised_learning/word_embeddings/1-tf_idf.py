#!/usr/bin/env python3
"""
Module containing the tf_idf function.
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix with L2 normalization.

    Parameters:
    sentences (list): A list of sentences to analyze.
    vocab (list): A list of vocabulary words to use for the analysis.
                  If None, all words within sentences are used.

    Returns:
    embeddings (numpy.ndarray): Matrix of shape (s, f) with TF-IDF values.
    features (list): Sorted list of the features used for the embeddings.
    """
    cleaned_sentences = []

    # Preprocess text identical to Task 0
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

    # Step 1: Compute Term Frequency (TF) -> raw count of term in document
    tf = np.zeros((s, f))
    feature_indices = {word: i for i, word in enumerate(features)}

    for row_idx, words in enumerate(cleaned_sentences):
        for word in words:
            if word in feature_indices:
                col_idx = feature_indices[word]
                tf[row_idx, col_idx] += 1

    # Step 2: Compute Inverse Document Frequency (IDF)
    # Number of documents containing each specific term
    doc_counts = np.sum(tf > 0, axis=0)

    # Use log(Total Documents / Documents with Term) matching standard setups
    # To protect against division by zero for terms not in corpus, cap doc count
    idf = np.zeros(f)
    for i in range(f):
        if doc_counts[i] > 0:
            idf[i] = np.log(s / doc_counts[i])
        else:
            idf[i] = 0.0

    # Step 3: Compute raw TF-IDF matrix
    embeddings = tf * idf

    # Step 4: Apply L2 (Euclidean) row-normalization
    # Sum of squares along each row axis
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Divide by norm where norm is greater than zero to avoid NaN values
    embeddings = np.divide(embeddings, norms, out=embeddings,
                           where=norms > 0)

    return embeddings, features
