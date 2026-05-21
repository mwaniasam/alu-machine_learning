# Natural Language Processing: Word Embeddings

## Description
This project explores various techniques in Natural Language Processing (NLP) for converting text data into meaningful numeric vectors (Word Embeddings). It covers classic frequency-based representations like Bag of Words (BoW) and TF-IDF, as well as modern static and contextual distributed representations including Word2Vec (CBOW and Skip-gram), GloVe, fastText, and ELMo.

## Environment & Requirements
*   **Operating System:** Ubuntu 16.04 LTS
*   **Compiler/Interpreter:** Python 3.5
*   **Core Libraries:**
    *   `numpy` (version 1.15)
    *   `tensorflow` (version 1.12)
    *   `gensim` (version 3.8.x)
    *   `keras` (version 2.2.5)
*   **Style Guide:** Code conforms to `pycodestyle` (version 2.4).
*   **Documentation:** All modules, classes, and functions contain descriptive `__doc__` strings.
*   **File Formatting:** All scripts begin with `#!/usr/bin/env python3`, are executable, and end with a trailing newline.

## Core Concepts & Learning Objectives

### 1. Natural Language Processing (NLP)
NLP is a field of artificial intelligence focused on enabling computers to understand, interpret, and manipulate human language. It bridges the gap between human communication and computer understanding.

### 2. Word Embeddings
Word embeddings are dense, low-dimensional vector representations of words where geometric closeness corresponds to semantic similarity. Unlike high-dimensional sparse vectors, embeddings capture contextual meaning and relationships between words.

### 3. Bag of Words (BoW)
A text representation technique that models text by counting the frequency of words within a document, completely ignoring grammar, word order, and syntax.

### 4. TF-IDF (Term Frequency-Inverse Document Frequency)
A statistical measure used to evaluate how important a word is to a document within a collection or corpus.
*   **Term Frequency (TF):** Measures how frequently a term occurs in a single document.
*   **Inverse Document Frequency (IDF):** Reduces the weight of terms that appear very frequently across all documents (like "the", "is"), highlighting unique, informational words.

### 5. Word2Vec Architectures
*   **Continuous Bag-of-Words (CBOW):** A neural network model that predicts a target word based on its surrounding context words.
*   **Skip-gram:** A neural network model that uses a single target word to predict its surrounding context words.
*   **Negative Sampling:** An efficient training optimization technique for Word2Vec that updates only a small percentage of negative weights instead of the entire vocabulary on each iteration.

### 6. n-gram
A contiguous sequence of $n$ items (characters or words) from a given sample of text.

### 7. Modern Embedding Paradigms
*   **GloVe (Global Vectors for Word Representation):** An unsupervised learning algorithm that builds word embeddings by performing matrix factorization on a global word-word log-co-occurrence matrix.
*   **fastText:** An extension of Word2Vec developed by Facebook that represents each word as a bag of character n-grams. This allows it to capture internal word morphology and generate embeddings for out-of-vocabulary words.
*   **ELMo (Embeddings from Language Models):** A deeply contextualized word representation model that models both complex characteristics of word use (syntax and semantics) and how these uses vary across linguistic contexts using deep, bi-directional LSTMs.

---

## File Map & Tasks

| File | Description |
| --- | --- |
| `0-bag_of_words.py` | Implementation of a Bag of Words matrix constructor. |
| `1-tf_idf.py` | Implementation of a TF-IDF matrix constructor. |
| `2-word2vec.py` | Implementation of a Word2Vec model trainer using Gensim. |
| `3-gensim_to_keras.py` | Integration script converting Gensim vectors to a Keras Embedding Layer. |

