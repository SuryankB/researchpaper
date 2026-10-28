
import pandas as pd
import unicodedata
from pathlib import Path

# Input and output files
input_file = Path("results/gemma_qwen_comparison_per_example.csv")
output_file = Path("results/consistent_word_counts_per_example.csv")

# Load the per-article comparison data
df = pd.read_csv(input_file)

# Check that all required columns exist
required_columns = [
    "language",
    "id",
    "human_summary",
    "gemma_summary",
    "qwen_summary",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Count words consistently:
# normalize Unicode to NFC, strip outer whitespace, split on whitespace.
def count_words(value):
    if pd.isna(value):
        return 0
    text = unicodedata.normalize("NFC", str(value)).strip()
    return len(text.split()) if text else 0

# Keep article identifiers and original summaries for traceability
result = df[["language", "id"]].copy()

result["reference_words"] = df["human_summary"].apply(count_words)
result["gemma_words"] = df["gemma_summary"].apply(count_words)
result["qwen_words"] = df["qwen_summary"].apply(count_words)

# Save the per-example counts
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved: {output_file.resolve()}")
print(f"Total rows: {len(result)}")
print("\nRows per language:")
print(result["language"].value_counts())

print("\nWord-count summary:")
print(
    result.groupby("language")[
        ["reference_words", "gemma_words", "qwen_words"]
    ].agg(["mean", "std"]).round(2)
)

print("\nFirst five rows:")
print(result.head().to_string(index=False))
