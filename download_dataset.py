"""Download the public ISOT Fake/Real News dataset into data/Fake.csv and data/True.csv.

The script tries a public GitHub mirror first, then the Kaggle dataset download endpoint.
Internet access is required. Review the dataset's license and limitations before reuse.
"""
from pathlib import Path
import io
import sys
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SOURCES = [
    # Public GitHub mirror containing Fake.csv and True.csv
    "https://github.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/archive/refs/heads/main.zip",
    "https://github.com/Reinforz/Fake-News-detection-with-ISOT-Dataset/archive/refs/heads/master.zip",
    # Kaggle's public dataset endpoint; may require an authenticated Kaggle setup in some environments.
    "https://www.kaggle.com/api/v1/datasets/download/clmentbisaillon/fake-and-real-news-dataset",
]


def extract_dataset(content: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.namelist()
            fake = next((m for m in members if Path(m).name.lower() == "fake.csv"), None)
            real = next((m for m in members if Path(m).name.lower() in {"true.csv", "real.csv"}), None)
            if not fake or not real:
                return False
            (DATA / "Fake.csv").write_bytes(archive.read(fake))
            (DATA / "True.csv").write_bytes(archive.read(real))
            return True
    except (zipfile.BadZipFile, OSError, KeyError):
        return False


for url in SOURCES:
    print(f"Trying dataset source: {url}")
    try:
        response = requests.get(url, timeout=(20, 180), stream=True, headers={"User-Agent": "FakeNewsDetectionProject/1.0"})
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                print(f"Downloaded {total / (1024 * 1024):.1f} MB", end="\r")
        content = b"".join(chunks)
        print()
        if extract_dataset(content):
            print("Dataset downloaded successfully:")
            print(f"  {DATA / 'Fake.csv'}")
            print(f"  {DATA / 'True.csv'}")
            sys.exit(0)
        print("This source did not contain both expected CSV files; trying next source.")
    except Exception as exc:
        print(f"Download failed: {exc}")

print("\nAutomatic download failed. Download Fake.csv and True.csv manually from the dataset page:")
print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
print("Place both files in the project's data/ folder, then run: python train_model.py")
sys.exit(1)
