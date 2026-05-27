# Transformer Applications - Machine Translation

This project focuses on building an end-to-end Machine Translation pipeline using the Transformer architecture. Instead of relying purely on higher-level abstraction libraries, this implementation covers data engineering, subword tokenization, custom text preprocessing pipelines, and a structured, vectorized training setup using TensorFlow.

---

## Core Project Concepts

### 1. Dataset & Pipeline Setup
The pipeline utilizes the structural dataset `ted_hrlr_translate/pt_to_en` provided by TensorFlow Datasets (TFDS). It sets up automated loading workflows for training and validation splits to feed tensors directly into our deep learning framework.

### 2. Subword Tokenization
Traditional word-level tokenization struggles with out-of-vocabulary (OOV) words, while character-level tokenization drastically increases sequence length. 
* **The Solution:** Subword tokenization dynamically segments text into variable-length units based on frequency.
* **Mechanism:** Using `SubwordTextEncoder`, common strings are grouped into a single unit, while rare terms are split safely into known subwords (e.g., "serendipity" $\rightarrow$ "seren", "dip", "ity"). This restricts vocabulary growth to a predefined threshold ($2^{15}$) without sacrificing semantic clarity.

### 3. Custom Training Workflows
Beyond typical model compilation blueprints, this project covers advanced Keras execution paradigms:
* Structuring efficient generator pipelines from streaming corpora.
* Custom processing of sequence boundaries via native Python execution bridges (`tf.py_function`).
* Implementing look-ahead and padding masks using targeted matrix operations (`tf.linalg.band_part`) to protect causal self-attention boundaries.

---

## Technical Architecture & Constraints

* **Platform Requirements:** Developed on Ubuntu 16.04 LTS utilizing Python 3.6.12.
* **Core Frameworks:** Executed using **TensorFlow 2.4.1** (imported via `tensorflow.compat.v2`) and **TensorFlow Datasets (TFDS)**.
* **Style Guidelines:** Source files comply strictly with the `pycodestyle` layout configuration (version 2.4).
* **Documentation Strategy:** Comprehensive docstrings are strictly enforced across all modules, classes, and helper functions to maintain codebase transparency and trace structural shapes.

---

## Technical Task Tracking

| Script File | Functional Description |
| --- | --- |
| `0-dataset.py` | Instantiates TFDS collection splits and builds isolated Portuguese/English vocabulary corpora using custom subword encoders. |
