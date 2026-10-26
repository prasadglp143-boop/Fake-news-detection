# Fake News Detection System — repaired version

This version removes the original synthetic-looking `data/news.csv` from the supplied project. The original file contained generic/random-looking text and could not support a meaningful fake-vs-real classifier. This package uses the public ISOT Fake and Real News dataset instead.

## Important accuracy note

The project cannot honestly promise a fixed accuracy before training on the dataset available on your machine. The training script measures accuracy on a stratified held-out 20% test split and saves the actual result in `models/metrics.json`. ISOT-style datasets often produce results above 85% with TF-IDF + Linear SVM, but this is not guaranteed and does not mean the model can fact-check every real-world claim. Source, topic, and writing-style artifacts can inflate benchmark results.

## Run in VS Code (Windows)

1. Extract the ZIP.
2. Open the `FakeNewsDetection` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, restart the terminal, and activate again. Alternatively use Command Prompt and `.venv\Scripts\activate.bat`.

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Download the dataset (internet required):

   ```powershell
   python download_dataset.py
   ```

   If automatic download fails, download the `Fake.csv` and `True.csv` files from the dataset page below and put them in `data/`.

   Dataset page: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

7. Check the dataset:

   ```powershell
   python check_data.py
   ```

8. Train the model and print the real test metrics:

   ```powershell
   python train_model.py
   ```

9. Start the web application:

   ```powershell
   python app.py
   ```

10. Open `http://127.0.0.1:5000` in your browser.

For terminal-only prediction, run `python predict.py` after training.

## Project files

- `download_dataset.py` — downloads the real labeled dataset.
- `train_model.py` — cleans data, deduplicates articles, trains TF-IDF + Linear SVM, evaluates on held-out test data, saves model and metrics.
- `text_utils.py` — shared text normalization for training and prediction.
- `app.py` — Flask web app; reads current saved evaluation metrics.
- `predict.py` — terminal prediction tool.
- `check_data.py` — data inspection helper.
- `data/` — place downloaded CSVs here.
- `models/` — generated after training.

## Labels and limitations

`Fake.csv` articles are labelled `fake`; `True.csv` articles are labelled `real`, following the dataset's published organization. A model prediction means that the text resembles patterns learned from this dataset. It is not a substitute for independent fact-checking, source verification, or human review.
