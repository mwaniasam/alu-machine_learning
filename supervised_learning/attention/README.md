# Attention Mechanisms and Transformer Architectures

This project delves deep into advanced Natural Language Processing (NLP) paradigms, specifically tracking the evolutionary leap from sequence-to-sequence (Seq2Seq) Recurrent Neural Networks (RNNs) with attention to the modern self-attention based Transformer architectures. 

The focus is on understanding, implementing, and utilizing these mathematical architectures natively within a deep learning framework to process sequence data efficiently.

---

## Core Project Concepts

### 1. Attention in Recurrent Neural Networks (RNNs)
Traditional Seq2Seq architectures compress an entire source sequence into a single, fixed-length bottleneck vector, causing significant information loss on longer sentences. 
* **The Solution:** The Attention mechanism enables the decoder network to look back at the full sequence of encoder hidden states at every generation step.
* **Mechanism:** It computes a dynamic alignment score (distribution) across all hidden states, allowing the model to focus on the specific input tokens most relevant to predicting the next target token.

### 2. The Transformer Architecture
Transformers eliminate recurrence entirely, opting instead to process entire sequences in parallel using **Self-Attention**. This removes the sequential computation bottleneck of RNNs and allows for massive parallelization during training.
* **Key Components:** Multi-Head Attention, Scaled Dot-Product Attention, Layer Normalization, Positional Encodings, and Position-Wise Feed-Forward Networks.

### 3. Pre-trained Language Frameworks
Beyond scratch-built architectures, this project explores foundational modern models that leverage self-supervised learning on massive corpora:
* **BERT (Bidirectional Encoder Representations from Transformers):** A deeply bidirectional transformer encoder framework designed to pre-train contextual word representations by masking tokens. It is highly versatile and easily fine-tuned for specialized downstream tasks.
* **GPT (Generative Pre-trained Transformer):** An autoregressive transformer decoder framework optimized for left-to-right causal language modeling and text generation.

---

## Environment & Technical Constraints

* **Platform:** Developed and tested on Ubuntu 16.04 LTS utilizing Python 3.5.
* **Framework Requirements:** Core models are constructed using **TensorFlow 1.15** and **NumPy 1.16**.
* **Coding Style:** All source files comply strictly with the `pycodestyle` compliance rules (version 2.4).
* **Documentation:** Comprehensive module, class, and function docstrings are enforced throughout to clarify tensor tracking, shapes, and structural logic.
