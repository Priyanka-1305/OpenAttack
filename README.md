# NLP Adversarial Attacks and Defenses Benchmark

This repository contains a comprehensive suite of experiments and evaluations for **Adversarial Attacks and Defense Mechanisms on Deep Learning NLP Classifiers**. The project utilizes the **OpenAttack** adversarial attack framework, along with Hugging Face **Transformers** (BERT, RoBERTa, and ELECTRA) to evaluate model robustness and benchmark various state-of-the-art text defense strategies.

---

## 🔍 Overview of the Project

The core goal of this project is to evaluate how modern NLP text classifiers perform under adversarial perturbations (both character-level and word-level) and to measure the effectiveness of defense mechanisms designed to mitigate these attacks. 

### 1. Victim Models
The project evaluates three popular transformer architectures:
*   **BERT** (`bert-base-uncased`)
*   **RoBERTa** (`roberta-base`)
*   **ELECTRA** (`google/electra-base-discriminator`)

### 2. Evaluated Adversarial Attacks (via OpenAttack)
*   **TextFooler (`TextFoolerAttacker`)**: A word-level synonym substitution attack that uses word embeddings to find semantic-preserving replacements.
*   **DeepWordBug (`DeepWordBugAttacker`)**: A character-level perturbation attack that introduces typos (insertions, deletions, substitutions, and swaps) to evade character-level spelling and word representations.
*   **UAT (`UATAttacker`)**: Universal Adversarial Triggers that search for a short sequence of words which, when prepended to any input text, triggers a target misclassification.
*   **SCPN (`SCPNAttacker`)**: Syntactically Controlled Paraphrase Networks that generate syntactic variations of input text to change the structure while keeping the meaning.
*   **GAN (`GANAttacker`)**: Generative Adversarial Network-based text perturbations.

### 3. Evaluated Defense Strategies
*   **Attention Drop (`Atttention_drop/`)**: A defense technique where attention weights are dropped out during training/inference to enhance the model's robustness against spelling and word perturbations.
*   **MaskPure (`bert_sst_maskpure_openattack.ipynb`)**: An implementation of the "Mask and Purify" defense framework. It masks highly vulnerable words and purifies/reconstructs the sentence using language models before passing it to the classifier.
*   **Imbalanced Training Defenses (`defense+imbalanced.ipynb`)**: Explores how models trained on imbalanced datasets behave under adversarial attacks and benches techniques to improve their robustness.

---

## 📊 Datasets Used
*   **SST-2 (Stanford Sentiment Treebank)**: Sentiment classification on movie review snippets.
*   **IMDB**: Binary sentiment classification on full-length movie reviews.
*   **Spam, Ham & Phishing**: Binary classification for identifying malicious/spam emails.
*   **Multiclass News**: Classification of news text into multiple distinct category labels.
*   **Hindi Translation Dataset (`dataset-merged.csv`)**: Evaluates cross-lingual or translation-based attacks and defenses on Hindi text.

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
