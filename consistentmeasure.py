import pandas as pd
import re

df = pd.read_csv("your_file.csv")

def word_count(text):
    if pd.isna(text):
        return 0
    text = str(text).strip()
    if not text:
        return 0
    return len(re.findall(r"\S+", text))

# Change these names to match your CSV
text_columns = {
    "reference": "reference",
    "gemma": "gemma_summary",
    "qwen": "qwen_summary"
}

for name, column in text_columns.items():
    df[f"{name}_word_count"] = df[column].apply(word_count)

print(df[[
    "reference_word_count",
    "gemma_word_count",
    "qwen_word_count"
]].mean())

# Save per-example lengths
df.to_csv("consistent_summary_lengths.csv", index=False)