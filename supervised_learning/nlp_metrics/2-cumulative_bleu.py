#!/usr/bin/env python3
"""
Module to calculate the cumulative n-gram BLEU score for a sentence.
"""
import numpy as np


def extract_ngrams(words_list, order):
    """
    Helper function to extract n-grams as tuples of words.
    """
    ngrams = []
    for i in range(len(words_list) - order + 1):
        ngrams.append(tuple(words_list[i:i + order]))
    return ngrams


def calculate_ngram_precision(references, sentence, order):
    """
    Helper function to calculate the modified precision for a specific order.
    """
    cand_ngrams = extract_ngrams(sentence, order)
    if not cand_ngrams:
        return 0.0

    cand_counts = {}
    for ngram in cand_ngrams:
        cand_counts[ngram] = cand_counts.get(ngram, 0) + 1

    max_ref_counts = {}
    for ngram in cand_counts:
        max_count = 0
        for ref in references:
            ref_ngrams = extract_ngrams(ref, order)
            count = ref_ngrams.count(ngram)
            if count > max_count:
                max_count = count
        max_ref_counts[ngram] = max_count

    clipped_counts = sum(min(cand_counts[ngram], max_ref_counts[ngram])
                         for ngram in cand_counts)

    return clipped_counts / len(cand_ngrams)


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Parameters:
    - references: list of reference translations
                  each reference translation is a list of words
    - sentence: list containing the model proposed sentence
    - n: the size of the largest n-gram to use for evaluation

    Returns:
    - the cumulative n-gram BLEU score
    """
    c = len(sentence)

    # Find the closest reference length 'r'
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(ref_lengths, key=lambda ref_len: (
        abs(ref_len - c), ref_len))

    precisions = []
    for i in range(1, n + 1):
        p_i = calculate_ngram_precision(references, sentence, i)
        precisions.append(p_i)

    # If any precision score is 0, the geometric mean component becomes 0
    if 0.0 in precisions:
        return 0.0

    # Calculate uniform weights
    weights = [1 / n] * n

    # Compute the geometric mean of precision scores
    geom_mean = np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))

    # Calculate Brevity Penalty (BP)
    if c > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c)

    return bp * geom_mean
