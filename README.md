# OpenAttack: An Open-Source Framework for Textual Adversarial Attack

This repository contains the code, notebooks, and benchmarks for our research project on adversarial robustness in NLP, focusing on the evaluation of baseline and novel attacks and defenses.

## 👥 Authors & Affiliation
*   **Nidhi M** (231AI024) — `nidhim.231ai024@nitk.edu.in`
*   **Priyanka Nitin Mohorikar** (231AI028) — `priyankanitinmohorikar.231ai028@nitk.edu.in`
*   **Varshini Reddy** (231AI041) — `varshinireddy.231ai041@nitk.edu.in`
*   **Advisor**: Dr. Anand Kumar M
*   **Institution**: Department of Information Technology, NITK Surathkal

---

## 🔍 Overview of the Project
We present a comprehensive adversarial robustness evaluation framework that integrates attack benchmarking, defense analysis, multilingual generalization, and a novel sentence-level adversarial attack named **TECA**.

### 1. The Novel TECA (Sentence-level) Attack
Unlike token-level perturbations (e.g., swapping words), **TECA** operates on syntactic and semantic transformations to construct semantically consistent but misleading sentences.
*   **Transformations**:
    *   Passive ↔ Active voice conversion
    *   Clause insertion or expansion
    *   Negation injection
    *   Phrase reordering
*   **Search Strategy**: Beam search is used to iteratively refine candidate sentences, optimizing the loss difference while enforcing a semantic similarity constraint ($Sim(x, x') \geq \delta$ via SBERT embeddings).

### 2. Baseline Attacks Evaluated
Using the **OpenAttack** toolkit, we benchmark:
*   **TextFooler**: Word-level synonym substitution.
*   **SCPN (Syntactically Controlled Paraphrase Networks)**: Syntactic structure perturbation.
*   **GAN-based Attacks**: Generative adversarial perturbations.
*   **DeepWordBug**: Character-level typos (used in defense analysis).
*   **UAT (Universal Adversarial Triggers)**: Trigger sequences (used in defense analysis).

### 3. Evaluated Defense Mechanisms
*   **MaskPure**: A stochastic purification layer applied during inference that randomly masks tokens and purifies them using a masked language model.
*   **AttentionDrop**: A transformer regularization technique that randomly drops attention weights during training to distribute token influence.
*   **Paraphrasing Defense**: A preprocessing step where a **T5** model generates 5 paraphrases of an input sentence. The model evaluates the 6 variants (5 paraphrases + 1 original) and averages the predictions.

---

## 📊 Datasets & Task Variants
We benchmarked on diverse datasets spanning binary, multiclass, imbalanced, and multilingual settings:
1.  **SST-2 (Binary Sentiment)**: Movie reviews with human annotations (70,042 entries).
2.  **IMDb Reviews (Binary Sentiment)**: Highly polar, long movie reviews (50,000 entries).
3.  **Mental Health Dataset (Imbalanced Multiclass)**: 127,241 entries with 13 emotion classes (where `Neutral` is 83%, other classes 13%, nulls removed).
4.  **Personality Dataset (Multiclass Classification)**: Kaggle Personality Dataset classifying text into Extrovert, Introvert, and Ambivert (10K–20K samples).
5.  **AG News (Multiclass Classification)**: Balanced dataset with World, Sports, Business, and Technology news (120,000 entries).
6.  **Hindi Dataset (Multilingual Evaluation)**: 15,000 sentiment examples to evaluate cross-lingual robustness.

---

## 📈 Key Results & Findings

### Defense Effectiveness (on SST-2 with BERT)
| Defense Model | TextFooler ASR | DeepWordBug ASR | UATA ASR |
| :--- | :---: | :---: | :---: |
| **Without Defense** | 0.81 | 0.52 | 0.09 |
| **MaskPure** | 0.42 | 0.32 | 0.09 |
| **AttentionDrop** | 0.64 | 0.46 | 0.09 |
| **Paraphrase Defense** | 0.76 | 0.29 | 0.09 |

### TECA vs. Baseline Attacks (ASR comparison)
*   **Near-Perfect Success**: TECA achieves near-perfect Attack Success Rate (ASR) of **0.96 to 1.00** across SST-2, IMDb, Imbalanced, and AG News datasets, consistently outperforming TextFooler, SCPN, and GANs.
*   **Personality Dataset Failure**: On the Personality dataset, TECA achieves **0.00 ASR** across all models (BERT, RoBERTa, ELECTRA) because TECA's syntactic transformations do not capture the subtle semantic markers required for personality classification.
*   **Multilingual (Hindi) Failure**: On the Hindi dataset, the ASR is **0.00**. This is due to Out-of-Vocabulary (OOV) tokens, tokenizer vocabulary mismatch, and model training bias towards English data.

---

## 📁 Repository Directory Structure

```bash
DL-project/
│
├── Atttention_drop/                 # Attention Drop defense experiments
│   ├── attack_attentiondrop.py      # Python script evaluating TextFooler on Attention Drop
│   └── attack_attentiondrop.ipynb   # Interactive notebook for testing Attention Drop
│
├── Post_midsem/                     # Post-midsem experiments and multiclass analysis
│   ├── SST2/                        # SST-2 RoBERTa/BERT/ELECTRA configurations
│   ├── imbalanced/                  # Imbalanced dataset training configurations
│   ├── imdb_/                       # IMDB dataset model configurations
│   ├── multiclass/                  # Multiclass model configurations
│   ├── multiclassnews/              # Multiclass news dataset configurations
│   ├── spam_ham_phishing_dataset.ipynb
│   ├── openattack_sst2.ipynb
│   ├── openattack_imdb.ipynb
│   ├── openattack_imbalanced.ipynb
│   ├── openattack_multiclasss.ipynb
│   ├── openattack_multiclasssnews.ipynb
│   └── test.csv
│
├── dataset-merged.csv               # Dataset used for Hindi attack experiments
├── hindi_dataset_attack.ipynb       # Adversarial attack evaluations on Hindi text
├── openattack_imdb.ipynb            # Attack evaluation notebook on IMDB dataset
├── defense+imbalanced.ipynb         # Defense experiments on imbalanced data
├── bert_sst_maskpure_openattack.ipynb # Notebook evaluating the MaskPure defense
├── Defence_Model.ipynb              # Baseline defense model notebook
│
├── .gitignore                       # Automatically ignores large binaries & datasets
└── README.md                        # Project documentation (this file)
```

> [!NOTE]
> Large weight checkpoints (`*.pt`, `*.safetensors`, `.jar` files) and large directories like `model/`, `model_imbalanced/`, and `data/` are stored locally on disk but ignored by `.gitignore` to keep the GitHub repository clean and lightweight.

---

## 🚀 How to Run the Experiments

### Prerequisites
Make sure you have `python 3.8+` installed. You will need to install the following dependencies:

```bash
pip install torch transformers datasets scikit-learn tqdm pandas OpenAttack
```

*Note: For SCPN and Stanford Parser based attacks, you may also need to download the Java-based Stanford Parser models and place them in the ignored `data/` directory.*

### Running an Evaluation
To evaluate the **Attention Drop** model against a `TextFooler` attack, run:

```bash
python Atttention_drop/attack_attentiondrop.py
```

To run individual notebook experiments:
1. Start Jupyter Notebook or open the workspace in VS Code.
2. Open any `.ipynb` file in `Post_midsem/` or the root folder.
3. Run the cells sequentially to load models, wrap them with the OpenAttack classifier, and evaluate adversarial attack success rates.
