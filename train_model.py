# train_model.py
# --------------  © 2025 --------------

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import os

DATA_FILE = "cleaned_data.txt"

# ---------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    cleaned_data.txt format:
        word<TAB>POS
        <blank line = sentence boundary>  ← ignored here
    Returns DataFrame[word, tag]
    """
    words, tags = [], []
    try:
        with open(path, "r", encoding="utf8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue               # skip sentence‑break line
                # Support both tab or space separation
                parts = ln.split("\t") if "\t" in ln else ln.split()
                if len(parts) != 2:
                    # skip malformed lines, but warn once
                    print(f"⚠️  Skipped malformed line: {ln[:40]}…")
                    continue
                word, tag = parts
                words.append(word)
                tags.append(tag)
    except FileNotFoundError:
        print(f"❌  File '{path}' not found. Make sure it is in the same folder.")
        sys.exit(1)

    df = pd.DataFrame({"word": words, "tag": tags})
    if df.empty:
        print("❌  Dataset empty ― check file content.")
        sys.exit(1)
    return df


# ---------------------------------------------------------
def train_and_save(df: pd.DataFrame):
    """Train Linear SVM, evaluate, save artefacts"""
    # Label‑encode words & tags
    w_enc, t_enc = LabelEncoder(), LabelEncoder()
    X = w_enc.fit_transform(df.word).reshape(-1, 1)
    y = t_enc.fit_transform(df.tag)

    Xtr, Xts, ytr, yts = train_test_split(
    X, y, test_size=0.20, random_state=42
)


    print(f"🔢  Train size: {len(ytr)},  Test size: {len(yts)}")

    # ------------  Model ------------
    clf = LinearSVC()        # good default for many classes
    clf.fit(Xtr, ytr)

    # ------------  Evaluation ------------
    ypred = clf.predict(Xts)
    acc = accuracy_score(yts, ypred) * 100
    print(f"\n✅  Accuracy: {acc:.2f}% on held‑out test set\n")
    # detailed report (handles unseen classes automatically)
    print(classification_report(
        yts, ypred,
        labels=list(range(len(t_enc.classes_))),
        target_names=t_enc.classes_,
        digits=3,
        zero_division=0
    ))

    # ------------  Save artefacts ------------
    joblib.dump(clf,       "model.pkl")
    joblib.dump(w_enc,     "word_enc.pkl")
    joblib.dump(t_enc,     "tag_enc.pkl")
    print("\n💾  Saved: model.pkl  word_enc.pkl  tag_enc.pkl")

# ---------------------------------------------------------
if __name__ == "__main__":
    print("📂  Loading dataset…")
    data = load_dataset(DATA_FILE)
    print(f"📄  Loaded {len(data)} word‑tag pairs.\n")
    train_and_save(data)
