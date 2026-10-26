"""Train and evaluate a TF-IDF + Linear SVM fake/real news classifier.

Expected data files in data/: Fake.csv and True.csv (ISOT format), or news.csv
with title/text/label columns. Metrics are measured on a held-out test split.
"""
from pathlib import Path
import json
import re
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from text_utils import clean_text

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset():
    unified = DATA_DIR / "news.csv"
    fake_path, real_path = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
    if unified.exists():
        df = pd.read_csv(unified)
        cols = {str(c).strip().lower(): c for c in df.columns}
        title_col = cols.get("title")
        text_col = cols.get("text") or cols.get("content")
        label_col = cols.get("label")
        if not text_col or not label_col:
            raise ValueError("news.csv must contain at least 'text' and 'label' columns; 'title' is optional.")
        titles = df[title_col].fillna("").astype(str) if title_col else ""
        df["content"] = (titles + " " + df[text_col].fillna("").astype(str)).str.strip()
        df["label_norm"] = df[label_col].astype(str).str.strip().str.lower()
        mapping = {"fake": "fake", "false": "fake", "0": "fake", "1": "real", "real": "real", "true": "real"}
        df["label"] = df["label_norm"].map(mapping)
        df = df[df["label"].notna()].copy()
    elif fake_path.exists() and real_path.exists():
        fake = pd.read_csv(fake_path, on_bad_lines="skip", low_memory=False)
        real = pd.read_csv(real_path, on_bad_lines="skip", low_memory=False)
        # Standard ISOT column names are title, text, subject, date (case-insensitive).
        fake.columns = [str(c).strip().lower() for c in fake.columns]
        real.columns = [str(c).strip().lower() for c in real.columns]
        for df in (fake, real):
            if "title" not in df.columns or "text" not in df.columns:
                raise ValueError("Fake.csv and True.csv must contain 'title' and 'text' columns.")
        fake = fake[["title", "text"]].copy()
        fake["label"] = "fake"
        real = real[["title", "text"]].copy()
        real["label"] = "real"
        df = pd.concat([fake, real], ignore_index=True)
        df["content"] = (df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)).str.strip()
    else:
        raise FileNotFoundError(
            "Dataset not found. Run 'python download_dataset.py' first, or put Fake.csv and True.csv in data/."
        )

    df["content"] = df["content"].map(clean_text)
    df = df[df["content"].str.split().str.len().ge(8)].copy()
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both fake and real labels. Check label values.")
    if df["label"].value_counts().min() < 10:
        raise ValueError("Not enough examples in one of the classes to train a useful model.")
    return df


def main():
    print("Loading and validating dataset...")
    data = load_dataset()
    print(f"Usable articles: {len(data):,}")
    print("Class counts:")
    print(data["label"].value_counts().to_string())
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.20, random_state=42, stratify=data["label"]
    )
    print(f"\nTraining rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=2, max_df=0.98, max_features=180000, sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Training Linear SVM...")
    model = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
        "precision": round(precision_score(y_test, predictions, pos_label="real", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, predictions, pos_label="real", zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, predictions, pos_label="real", zero_division=0) * 100, 2),
        "best_model": "Linear SVM",
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_counts": {str(k): int(v) for k, v in data["label"].value_counts().items()},
        "test_split": "stratified 80/20, random_state=42",
        "note": "Benchmark performance on this dataset is not proof of real-world truth verification. Publisher/topic artifacts may affect scores."
    }
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["fake", "real"], zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; fake, real):")
    print(confusion_matrix(y_test, predictions, labels=["fake", "real"]))

    joblib.dump(model, MODEL_DIR / "fake_news_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved model and metrics to: {MODEL_DIR}")
    if metrics["accuracy"] < 85:
        print("WARNING: Accuracy is below 85%. Do not report an 85% result; inspect labels, data quality, and split leakage.")
    else:
        print(f"Accuracy is {metrics['accuracy']}% on this held-out split (not a guarantee for new sources).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
