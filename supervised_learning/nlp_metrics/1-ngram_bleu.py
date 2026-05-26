#!/usr/bin/env python3
"""
Module to calculate the n-gram BLEU score for a sentence.
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Parameters:
    - references: list of reference translations
                  each reference translation is a list of words
    - sentence: list containing the model proposed sentence
    - n: the size of the n-gram to use for evaluation

    Returns:
    - the n-gram BLEU score
    """
    c = len(sentence)

    # Find the closest reference length 'r'
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(ref_lengths, key=lambda ref_len: (
        abs(ref_len - c), ref_len))

    # Helper lambda to extract n-grams as tuples of words
    def extract_ngrams(words_list, order):
        ngrams = []
        for i in range(len(words_list) - order + 1):
            ngrams.append(tuple(words_list[i:i + order]))
        return ngrams

    # Extract n-grams for the candidate sentence
    cand_ngrams = extract_ngrams(sentence, n)
    if not cand_ngrams:
        return 0.0

    # Count frequencies of n-grams in candidate sentence
    cand_counts = {}
    for ngram in cand_ngrams:
        cand_counts[ngram] = cand_counts.get(ngram, 0) + 1

    # Find the maximum frequency of each n-gram across any single reference
    max_ref_counts = {}
    for ngram in cand_counts:
        max_count = 0
        for ref in references:
            ref_ngrams = extract_ngrams(ref, n)
            count = ref_ngrams.count(ngram)
            if count > max_count:
                max_count = count
        max_ref_counts[ngram] = max_count

    # Calculate clipped counts for modified precision
    clipped_counts = sum(min(cand_counts[ngram], max_ref_counts[ngram])
                         for ngram in cand_counts)

    precision = clipped_counts / len(cand_ngrams)

    # If precision is 0, the overall BLEU score is 0
    if precision == 0:
        return 0.0

    # Calculate Brevity Penalty (BP)
    if c > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c)

    return bp * precision
