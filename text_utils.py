"""Text normalization shared by training and prediction."""
import re
import html


def clean_text(value) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    # Remove the highly dataset-specific Reuters wire-service dateline marker.
    text = re.sub(r"^\s*\(?\s*reuters\s*\)?\s*[-:,.]?\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
