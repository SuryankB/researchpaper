
import os
import unicodedata

# Force Hugging Face libraries to use cached files only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "qwen_generations_checkpoint.csv"
)
PER_EXAMPLE_CSV = os.path.join(
    PROJECT_DIR, "results", "qwen_evaluation_per_example.csv"
)
LANGUAGE_CSV = os.path.join(
    PROJECT_DIR, "results", "qwen_evaluation_by_language.csv"
)
LENGTH_CSV = os.path.join(
    PROJECT_DIR, "results", "qwen_summary_length_analysis.csv"
)


class UnicodeTokenizer:
    """Tokenizer that preserves Unicode letters, marks, and numbers."""

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


def main():
    df = pd.read_csv(INPUT_CSV).fillna("")
    df = df[df["status"] == "success"].copy()

    required = {"language", "id", "human_summary", "qwen_summary"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != 400:
        raise ValueError(
            f"Expected 400 successful rows; found {len(df)}"
        )

    # ROUGE: Unicode-aware tokenization for Hindi and Bengali.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=UnicodeTokenizer(),
    )

    rouge_results = []

    for _, row in df.iterrows():
        reference = str(row["human_summary"])
        prediction = str(row["qwen_summary"])

        # ROUGE expects target/reference first and prediction second.
        scores = scorer.score(reference, prediction)

        rouge_results.append({
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure,
        })

    metrics = pd.DataFrame(rouge_results, index=df.index)
    df = pd.concat([df, metrics], axis=1)

    # Save ROUGE results before BERTScore, so they survive a later error.
    df.to_csv(
        PER_EXAMPLE_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    print("Unicode-aware ROUGE results saved; proceeding to BERTScore.")

    # BERTScore: retain the existing model and language settings.
    df["bertscore_precision"] = np.nan
    df["bertscore_recall"] = np.nan
    df["bertscore_f1"] = np.nan

    for language in ["Hindi", "Bengali"]:
        idx = df.index[
            df["language"].astype(str).str.lower() == language.lower()
        ].tolist()

        if not idx:
            continue

        predictions = (
            df.loc[idx, "qwen_summary"].astype(str).tolist()
        )
        references = (
            df.loc[idx, "human_summary"].astype(str).tolist()
        )

        P, R, F1 = bert_score(
            predictions,
            references,
            lang="hi" if language == "Hindi" else "bn",
            model_type="bert-base-multilingual-cased",
            verbose=True,
        )

        df.loc[idx, "bertscore_precision"] = P.cpu().numpy()
        df.loc[idx, "bertscore_recall"] = R.cpu().numpy()
        df.loc[idx, "bertscore_f1"] = F1.cpu().numpy()

    # Save per-example results after both evaluations.
    df.to_csv(
        PER_EXAMPLE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # Aggregate means, sample standard deviations, and counts.
    metric_cols = [
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]

    summary = df.groupby("language")[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary_output = summary.reset_index()
    summary_output.to_csv(
        LANGUAGE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # Length analysis: whitespace-separated tokens, as before.
    df["human_length_tokens"] = (
        df["human_summary"].astype(str).str.split().str.len()
    )
    df["qwen_length_tokens"] = (
        df["qwen_summary"].astype(str).str.split().str.len()
    )
    df["length_difference_tokens"] = (
        df["qwen_length_tokens"] - df["human_length_tokens"]
    )

    length_summary = df.groupby("language").agg(
        n=("id", "count"),
        human_length_mean=("human_length_tokens", "mean"),
        qwen_length_mean=("qwen_length_tokens", "mean"),
        length_difference_mean=("length_difference_tokens", "mean"),
        length_difference_median=("length_difference_tokens", "median"),
    ).reset_index()

    length_summary.to_csv(
        LENGTH_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEvaluation complete.")
    print("Rows evaluated:", len(df))
    print("\nScores by language:")
    print(summary_output.to_string(index=False))
    print("\nSummary length:")
    print(length_summary.to_string(index=False))
    print("\nSaved:")
    print(PER_EXAMPLE_CSV)
    print(LANGUAGE_CSV)
    print(LENGTH_CSV)


if __name__ == "__main__":
    main()
