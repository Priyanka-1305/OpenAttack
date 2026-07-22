# OpenAttack: Adversarial Attacks and Defenses on Text Classifiers

This repository contains the code, notebooks, and benchmarks for evaluating adversarial robustness in Natural Language Processing (NLP) models. Using the **OpenAttack** framework and Hugging Face **Transformers**, this project systematically benchmarks various baseline attacks, evaluates advanced defense strategies, and proposes a novel sentence-level adversarial attack.

---

## 🔍 Project Summary
The project assesses the vulnerability of state-of-the-art transformer classifiers to both token-level and sentence-level adversarial perturbations. It analyzes how different architectural designs (BERT, RoBERTa, and ELECTRA) behave under stress and evaluates the effectiveness of pre-processing, regularization, and purification defenses in mitigating adversarial success rates.

---

## 🧪 Research Overview

### 1. Proposed Sentence-Level Attack: TECA
The project proposes **TECA**, a structured, sentence-level adversarial generation mechanism. Unlike token-swapping baselines, TECA generates semantically consistent but highly misleading sentences by applying:
*   Passive ↔ Active voice conversion
*   Clause insertion or expansion
*   Negation injection
*   Phrase reordering
*   **Search Optimization**: A beam search strategy is used to iteratively refine candidate sentences, maximizing the victim model's loss difference while maintaining a semantic similarity constraint ($Sim(x, x') \geq \delta$) calculated via SBERT embeddings.

### 2. Baseline Attacks Evaluated
*   **TextFooler**: Word-level synonym substitution.
*   **SCPN (Syntactically Controlled Paraphrase Networks)**: Syntactic structure perturbation.
*   **GAN-based Attacks**: Generative adversarial perturbations.
*   **DeepWordBug**: Character-level spelling perturbations (typos).
*   **UAT (Universal Adversarial Triggers)**: Short trigger sequences prepended to inputs.

### 3. Evaluated Defense Mechanisms
*   **MaskPure**: A stochastic purification layer applied during inference that randomly masks tokens and purifies them using a masked language model.
*   **AttentionDrop**: A transformer regularization technique that randomly drops attention weights during training to distribute token influence.
*   **Paraphrasing Defense**: A pre-processing step where a **T5** model generates 5 paraphrases of an input sentence. The model evaluates the 6 variants (5 paraphrases + 1 original) and averages the predictions.

---

## 📊 Datasets & Task Variants
Evaluations were conducted across a wide range of text classification paradigms:
1.  **SST-2 (Binary Sentiment)**: Stanford Sentiment Treebank (70,042 entries).
2.  **IMDb Reviews (Binary Sentiment)**: Movie reviews with long-form text (50,000 entries).
3.  **Mental Health Dataset (Imbalanced Multiclass)**: 127,241 entries with 13 emotion classes (where `Neutral` is 83% and other classes are 13%).
4.  **Personality Dataset (Multiclass)**: Kaggle Personality Dataset classifying text into Extrovert, Introvert, and Ambivert (10K–20K samples).
5.  **AG News (Multiclass)**: Balanced news dataset with 4 categories (120,000 entries).
6.  **Hindi Dataset (Multilingual)**: 15,000 sentiment examples to evaluate cross-lingual robustness.

---

## 📈 Key Results & Findings

### Defense Effectiveness (on SST-2 with BERT)
| Defense Model | TextFooler ASR | DeepWordBug ASR | UATA ASR |
| :--- | :---: | :---: | :---: |
| **Without Defense** | 0.81 | 0.52 | 0.09 |
| **MaskPure** | 0.42 | 0.32 | 0.09 |
| **AttentionDrop** | 0.64 | 0.46 | 0.09 |
| **Paraphrase Defense** | 0.76 | 0.29 | 0.09 |

### TECA Success and Failure Modes
*   **Near-Perfect Success**: TECA achieves near-perfect Attack Success Rate (ASR) of **0.96 to 1.00** across SST-2, IMDb, Imbalanced, and AG News datasets, consistently outperforming token-level attacks.
*   **Personality Dataset Failure**: On the Personality dataset, TECA achieves **0.00 ASR** across all victim models. Syntactic transformations fail to capture the nuanced semantic markers required for personality classification.
*   **Multilingual (Hindi) Failure**: On the Hindi dataset, the ASR is **0.00** across all samples. This failure is due to Out-of-Vocabulary (OOV) Hindi tokens, tokenizer vocabulary mismatch, and model training bias towards English data.

---

## 🚀 Getting Started

### Prerequisites
Install the required packages:
```bash
pip install torch transformers datasets scikit-learn tqdm pandas OpenAttack
```

### Running an Evaluation
To evaluate the **Attention Drop** model against a `TextFooler` attack, run:
```bash
python Atttention_drop/attack_attentiondrop.py
```
