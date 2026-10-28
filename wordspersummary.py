
from pathlib import Path
import pandas as pd
import re
import unicodedata

# Run this script from the folder containing your project CSV files.
ROOT = Path(".")

FILES = {
    "comparison": ROOT / "results/gemma_qwen_comparison_per_example.csv",
    "gemma": ROOT / "results/gemma_evaluation_per_example.csv",
    "qwen": ROOT / "results/qwen_evaluation_per_example.csv",
    "dataset": ROOT / "results/xlsum_sample_200_per_language.csv",
}

def word_count(value):
    """Count whitespace-separated words consistently."""
    if pd.isna(value):
        return 0
    text = unicodedata.normalize("NFC", str(value)).strip()
    return len(text.split()) if text else 0

def load_csv(path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n{path.name}: {len(df)} rows")
    print("Columns:", df.columns.tolist())
    return df

# First inspect all available files.
data = {name: load_csv(path) for name, path in FILES.items()}

print(
    "\nCheck the printed column names. "
    "Set the source column mapping below to match your files."
)

# This inspection-only script does not write an output yet.
# We need the actual column names and matching article IDs/language
# before safely joining reference, Gemma, and Qwen word counts.
