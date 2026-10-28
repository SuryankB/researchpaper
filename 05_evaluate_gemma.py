
import os

# Force Hugging Face libraries to use cached files only.
# Set these BEFORE importing bert_score or transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ---------- Files ----------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results"

INPUT_FILE = RESULTS / "gemma_generations_checkpoint.csv"

PER_EXAMPLE_FILE = RESULTS / "gemma_evaluation_per_example.csv"
SUMMARY_FILE = RESULTS / "gemma_evaluation_by_language.csv"
LENGTH_FILE = RESULTS / "gemma_summary_length_analysis.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {INPUT_FILE.resolve()}"
    )


# ---------- Load generated summaries ----------
df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    dtype={"id": str},
)

df = df[df["status"] == "success"].copy()

required = {
    "language",
    "id",
    "article",
    "human_summary",
    "gemma_summary",
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for column in ["article", "human_summary", "gemma_summary"]:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# Keep only rows with both a reference and a generated summary.
df = df[
    df["human_summary"].ne("")
    & df["gemma_summary"].ne("")
].copy()

if df.empty:
    raise ValueError("No valid summaries to evaluate.")

print("Examples to evaluate:", len(df))
print(df.groupby("language").size())


# ---------- Unicode-aware ROUGE tokenizer ----------
class UnicodeTokenizer:
    """Preserve Unicode letters, combining marks, and numbers."""

    def tokenize(self, text):
        text = unicodedata.normalize("NFC", str(text)).lower()

        tokens = []
        current = []

        for char in text:
            category = unicodedata.category(char)

            # L = letters, M = combining marks, N = numbers
            if category[0] in ("L", "M", "N"):
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []

        if current:
            tokens.append("".join(current))

        return tokens


tokenizer = UnicodeTokenizer()

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=tokenizer,
)


# ---------- ROUGE and length metrics ----------
rows = []

for _, row in df.iterrows():
    reference = str(row["human_summary"])
    candidate = str(row["gemma_summary"])

    scores = scorer.score(reference, candidate)

    ref_words = tokenizer.tokenize(reference)
    gen_words = tokenizer.tokenize(candidate)

    rows.append({
        "language": row["language"],
        "id": row["id"],

        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_f1": scores["rouge1"].fmeasure,

        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_f1": scores["rouge2"].fmeasure,

        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_f1": scores["rougeL"].fmeasure,

        "human_word_count": len(ref_words),
        "gemma_word_count": len(gen_words),

        "human_char_count": len(reference),
        "gemma_char_count": len(candidate),

        "human_sentence_count": len([
            s for s in re.split(r"[.!?।]+", reference)
            if s.strip()
        ]),
        "gemma_sentence_count": len([
            s for s in re.split(r"[.!?।]+", candidate)
            if s.strip()
        ]),
    })

metrics = pd.DataFrame(rows)

# Save ROUGE results before BERTScore in case it fails.
metrics.to_csv(
    PER_EXAMPLE_FILE,
    index=False,
    encoding="utf-8-sig",
)
print("\nUnicode-aware ROUGE results saved; proceeding to BERTScore.")


# ---------- Multilingual BERTScore ----------
print("\nCalculating multilingual BERTScore in offline mode...")
print("Required model and tokenizer files must already be cached.")

metrics["bertscore_precision"] = float("nan")
metrics["bertscore_recall"] = float("nan")
metrics["bertscore_f1"] = float("nan")

# Evaluate languages separately, matching the Qwen evaluation.
for language in ["Hindi", "Bengali"]:
    idx = metrics.index[
        metrics["language"].astype(str).str.lower()
        == language.lower()
    ].tolist()

    if not idx:
        continue

    predictions = (
        df.loc[idx, "gemma_summary"].astype(str).tolist()
    )
    references = (
        df.loc[idx, "human_summary"].astype(str).tolist()
    )

    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        lang="hi" if language == "Hindi" else "bn",
        model_type="bert-base-multilingual-cased",
        verbose=True,
    )

    metrics.loc[idx, "bertscore_precision"] = P.cpu().numpy()
    metrics.loc[idx, "bertscore_recall"] = R.cpu().numpy()
    metrics.loc[idx, "bertscore_f1"] = F1.cpu().numpy()

# Save completed per-example evaluation.
metrics.to_csv(
    PER_EXAMPLE_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ---------- Aggregate by language ----------
metric_columns = [
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
]

summary = metrics.groupby("language")[metric_columns].agg(
    ["mean", "std", "count"]
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ---------- Length analysis ----------
length_columns = [
    "human_word_count",
    "gemma_word_count",
    "human_char_count",
    "gemma_char_count",
    "human_sentence_count",
    "gemma_sentence_count",
]

length_rows = []

for language, group in metrics.groupby("language"):
    for column in length_columns:
        length_rows.append({
            "language": language,
            "measure": column,
            "mean": group[column].mean(),
            "std": group[column].std(ddof=1),
            "median": group[column].median(),
        })

length_summary = pd.DataFrame(length_rows)

length_summary.to_csv(
    LENGTH_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ---------- Final report ----------
print("\nEvaluation complete.")
print("Rows evaluated:", len(metrics))

print("\nROUGE and BERTScore averages by language:")

for _, row in summary.iterrows():
    print(f"\n{row['language']}")

    for metric in metric_columns:
        print(
            f"  {metric}: "
            f"{row[f'{metric}_mean']:.4f} "
            f"+/- {row[f'{metric}_std']:.4f}"
        )

print("\nSaved files:")
print(PER_EXAMPLE_FILE.resolve())
print(SUMMARY_FILE.resolve())
print(LENGTH_FILE.resolve())
