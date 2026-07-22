import torch
import OpenAttack as oa
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
# from OpenAttack.utils import DataInstance
# 1. Imports
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import pandas as pd
import OpenAttack as oa
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import Dataset, DatasetDict
from OpenAttack.tags import TAG_English



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Load model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

state_dict = torch.load("attentiondrop_sst2.pt", map_location=device)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# -----------------------------
# OpenAttack Victim Wrapper
# -----------------------------
class MyClassifier(oa.Classifier):

    def get_pred(self, inputs):
        tokens = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()

    def get_prob(self, inputs):
        tokens = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        probs = torch.softmax(outputs.logits, dim=1)

        return probs.cpu().numpy()

victim = MyClassifier()

# -----------------------------
# Load SST2 dataset
# -----------------------------

dataset = load_dataset("glue", "sst2")

samples = []

for i in range(50):
    text = dataset["validation"][i]["sentence"]
    label = int(dataset["validation"][i]["label"])

    samples.append({
        "x": text,
        "y": label
    })

# -----------------------------
# Choose Attack Method
# -----------------------------
attacker = oa.attackers.TextFoolerAttacker()

# -----------------------------
# Attack Evaluation
# -----------------------------
attack_eval = oa.AttackEval(
    attacker,
    victim
)

attack_eval.eval(samples)