from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
for name in ("Fake.csv", "True.csv", "news.csv"):
    path = ROOT / "data" / name
    if path.exists():
        print(f"\n{name}")
        try:
            df = pd.read_csv(path, nrows=10_000, on_bad_lines="skip", low_memory=False)
            print("Sample shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print(df.head(3).to_string(index=False, max_colwidth=100))
        except Exception as exc:
            print("Could not inspect:", exc)
if not any((ROOT / "data" / n).exists() for n in ("Fake.csv", "True.csv", "news.csv")):
    print("No dataset files found. Run: python download_dataset.py")
