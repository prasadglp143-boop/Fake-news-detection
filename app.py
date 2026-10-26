from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request
from text_utils import clean_text

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MODEL_PATH = ROOT / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

model = vectorizer = metrics = None
if MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METRICS_PATH.exists():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)


def get_counts():
    data_dir = ROOT / "data"
    unified = data_dir / "news.csv"
    if unified.exists():
        data = pd.read_csv(unified)
        label_col = next((c for c in data.columns if str(c).lower() == "label"), None)
        if label_col:
            labels = data[label_col].astype(str).str.lower()
            return len(data), int(labels.isin(["real", "true", "1"]).sum()), int(labels.isin(["fake", "false", "0"]).sum())
    counts = {"real": 0, "fake": 0}
    total = 0
    for filename, label in (("True.csv", "real"), ("Fake.csv", "fake")):
        path = data_dir / filename
        if path.exists():
            try:
                n = len(pd.read_csv(path, usecols=lambda c: str(c).lower() in {"title", "text"}, on_bad_lines="skip"))
            except Exception:
                n = 0
            total += n
            counts[label] += n
    return total, counts["real"], counts["fake"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    news = ""
    error = ""
    if request.method == "POST":
        news = request.form.get("news", "").strip()
        if model is None or vectorizer is None:
            error = "Model not trained yet. Run python train_model.py first."
        elif len(news.split()) < 8:
            error = "Please enter a full headline or article (at least 8 words) for a more meaningful prediction."
        else:
            cleaned = clean_text(news)
            result = model.predict(vectorizer.transform([cleaned]))[0]
            prediction = "REAL NEWS (matches the real-news training pattern)" if result == "real" else "FAKE NEWS (matches the fake-news training pattern)"
    total_news, real_news, fake_news = get_counts()
    display_metrics = metrics or {"accuracy": "—", "precision": "—", "recall": "—", "f1": "—", "best_model": "Not trained"}
    return render_template("index.html", prediction=prediction, news=news, error=error,
        total_news=total_news, real_news=real_news, fake_news=fake_news,
        accuracy=display_metrics.get("accuracy", "—"), precision=display_metrics.get("precision", "—"),
        recall=display_metrics.get("recall", "—"), f1=display_metrics.get("f1", "—"),
        best_model=display_metrics.get("best_model", "Not trained"))


if __name__ == "__main__":
    app.run(debug=False)
