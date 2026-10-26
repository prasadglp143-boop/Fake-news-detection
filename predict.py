from pathlib import Path
import joblib
from text_utils import clean_text

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "models" / "fake_news_model.pkl"
vectorizer_path = ROOT / "models" / "tfidf_vectorizer.pkl"
if not model_path.exists() or not vectorizer_path.exists():
    raise SystemExit("Model files are missing. Run: python train_model.py")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("=" * 48)
print("          FAKE NEWS DETECTION")
print("=" * 48)
news = input("Paste a full news headline/article: ").strip()
if len(news.split()) < 8:
    raise SystemExit("Please enter at least 8 words for a more meaningful prediction.")
result = model.predict(vectorizer.transform([clean_text(news)]))[0]
print("\nPrediction:", "REAL NEWS STYLE" if result == "real" else "FAKE NEWS STYLE")
print("Note: This model detects patterns learned from its dataset; it does not independently fact-check claims.")
