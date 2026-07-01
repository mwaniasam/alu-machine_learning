#!/usr/bin/env python3
"""
Semantic Search
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents
    """
    docs_text = []
    
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            with open(os.path.join(corpus_path, filename),
                      'r', encoding='utf-8') as f:
                docs_text.append(f.read())

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    embeddings = model([sentence] + docs_text)

    sentence_embedding = embeddings[0]
    docs_embeddings = embeddings[1:]

    correlation = np.inner(sentence_embedding, docs_embeddings)
    closest = np.argmax(correlation)

    return docs_text[closest]
