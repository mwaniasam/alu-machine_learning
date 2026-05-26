# Natural Language Processing - Evaluation Metrics

This project explores essential evaluation metrics used in Natural Language Processing (NLP) to assess the performance of language generation, translation, and text summarization models. It includes local implementations of metrics like BLEU without relying on high-level libraries like `nltk`.

## Learning Objectives

By the end of this project, you should be able to explain:
* The common real-world applications of Natural Language Processing.
* What a BLEU score is, how it is calculated, and its limitations.
* What a ROUGE score is and how it differs from BLEU.
* What perplexity represents in language modeling.
* When to choose one evaluation metric over another depending on the task.

## Requirements

* **Environment:** All files are interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5).
* **Dependencies:** Code is executed using `numpy` (version 1.15). The `nltk` module is strictly forbidden.
* **Style:** All source code complies with the `pycodestyle` style guide (version 2.4).
* **Documentation:** Modules, classes, and functions are comprehensively documented.
* **Executable:** All script files must be executable and start with `#!/usr/bin/env python3`.

---

## Core Metrics Overview

### 1. BLEU (Bilingual Evaluation Understudy)
* **Primary Use:** Machine Translation.
* **Mechanism:** Measures **precision** by calculating the overlap of n-grams (sequences of words) between the model's generated text and human-written reference translations. 
* **Brevity Penalty:** To prevent the model from generating artificially brief sentences to game precision metrics, a Brevity Penalty (BP) scaling factor is applied if the output is shorter than the reference.
* **Shortcomings:** It cannot judge structural meaning or semantic variations (synonyms), struggles with grammatical accuracy assessment, and behaves poorly with languages that lack concrete word boundaries.

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
* **Primary Use:** Document Summarization.
* **Mechanism:** Primarily measures **recall**. It assesses how many of the essential n-grams present in human reference summaries were successfully captured by the generated text.

### 3. Perplexity
* **Primary Use:** Evaluation of Language Models.
* **Mechanism:** Measures how well a probability distribution or language model predicts a sample text. Mathematically, it represents the geometric average number of word choices the model considers at any given step—known as the **branching factor**. Lower perplexity indicates a more confident and accurate model.

---
