#!/usr/bin/env python3
"""
Module to calculate the unigram BLEU score for a sentence.
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Parameters:
    - references: list of reference translations
                  each reference translation is a list of words
    - sentence: list containing the model proposed sentence

    Returns:
    - the unigram BLEU score
    """
    c = len(sentence)

    # Find the closest reference length 'r'
    ref_lengths = [len(ref) for ref in references]
    # In case of a tie, choose the shorter length
    closest_ref_len = min(ref_lengths, key=lambda ref_len: (
        abs(ref_len - c), ref_len))

    # Calculate Modified Unigram Precision
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    max_ref_counts = {}
    for word in word_counts:
        max_count = 0
        for ref in references:
            count = ref.count(word)
            if count > max_count:
                max_count = count
        max_ref_counts[word] = max_count

    clipped_counts = sum(min(word_counts[word], max_ref_counts[word])
                         for word in word_counts)

    precision = clipped_counts / c

    # Calculate Brevity Penalty (BP)
    if c > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c)

    return bp * precision
