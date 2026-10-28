
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "comparison_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

FILES = {
    "gemma_gen": RESULTS / "gemma_generations_checkpoint.csv",
    "qwen_gen": RESULTS / "qwen_generations_checkpoint.csv",
    "gemma_eval": RESULTS / "gemma_evaluation_per_example.csv",
    "qwen_eval": RESULTS / "qwen_evaluation_per_example.csv",
}

OUT_COMBINED = RESULTS / "gemma_qwen_comparison_per_example.csv"
OUT_SUMMARY = RESULTS / "gemma_qwen_comparison_by_language.csv"
OUT_LENGTH = RESULTS / "gemma_qwen_length_comparison.csv"
OUT_BOOTSTRAP = RESULTS / "gemma_qwen_paired_bootstrap.csv"

METRICS = [
    "rouge1_f1", "rouge2_f1", "rougeL_f1",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
]


# ---------- Shared Unicode tokenizer ----------
def unicode_tokens(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    tokens, current = [], []

    for char in text:
        if unicodedata.category(char)[0] in ("L", "M", "N"):
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tokens


def sentence_count(text):
    return len([
        s for s in re.split(r"[.!?।]+", str(text))
        if s.strip()
    ])


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})


def load_generation(path, model, summary_column):
    df = read_csv(path).fillna("")

    required = {
        "language", "id", "status", "human_summary", summary_column
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    df = df[df["status"].astype(str).str.lower() == "success"].copy()
    df["id"] = df["id"].astype(str)
    df["language"] = df["language"].astype(str).str.strip()
    df["human_summary"] = df["human_summary"].astype(str).str.strip()
    df[summary_column] = df[summary_column].astype(str).str.strip()

    df = df[
        df["human_summary"].ne("") & df[summary_column].ne("")
    ].copy()

    df = df.rename(columns={summary_column: f"{model}_summary"})
    return df[["language", "id", "human_summary", f"{model}_summary"]]


def load_evaluation(path, model):
    df = read_csv(path).fillna("")
    required = {"language", "id", *METRICS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}. "
            "Check that evaluation completed successfully."
        )

    df["id"] = df["id"].astype(str)
    df["language"] = df["language"].astype(str).str.strip()

    df = df[["language", "id", *METRICS]].copy()
    return df.rename(columns={
        metric: f"{model}_{metric}" for metric in METRICS
    })


def validate_pair(gemma, qwen):
    keys = ["language", "id"]

    if gemma.duplicated(keys).any() or qwen.duplicated(keys).any():
        raise ValueError("Duplicate language/article ID found.")

    g = gemma[keys].sort_values(keys).reset_index(drop=True)
    q = qwen[keys].sort_values(keys).reset_index(drop=True)

    if not g.equals(q):
        raise ValueError(
            "Gemma and Qwen do not have matching language/article IDs."
        )


def paired_bootstrap(gemma, qwen, n_boot=10000, seed=2026):
    """95% percentile CI for mean paired difference (Qwen - Gemma)."""
    g = np.asarray(gemma, dtype=float)
    q = np.asarray(qwen, dtype=float)
    valid = np.isfinite(g) & np.isfinite(q)
    differences = q[valid] - g[valid]

    n = len(differences)
    if n < 2:
        return n, np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)

    # Batch iterations to limit memory use.
    batch_size = 250
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        sample_indices = rng.integers(
            0, n, size=(stop - start, n)
        )
        boot_means[start:stop] = differences[sample_indices].mean(axis=1)

    low, high = np.quantile(boot_means, [0.025, 0.975])
    return n, differences.mean(), low, high


def main():
    # Load generated summaries.
    gemma_gen = load_generation(
        FILES["gemma_gen"], "gemma", "gemma_summary"
    )
    qwen_gen = load_generation(
        FILES["qwen_gen"], "qwen", "qwen_summary"
    )

    validate_pair(gemma_gen, qwen_gen)

    keys = ["language", "id"]

    combined = gemma_gen.merge(
        qwen_gen, on=keys, suffixes=("_gemma", "_qwen"),
        validate="one_to_one"
    )

    # The human reference must be identical for each paired article.
    if not (
        combined["human_summary_gemma"]
        == combined["human_summary_qwen"]
    ).all():
        raise ValueError("Reference summaries differ between models.")

    combined = combined.rename(
        columns={"human_summary_gemma": "human_summary"}
    )
    combined = combined.drop(columns=["human_summary_qwen"])

    # Load per-example evaluation metrics.
    gemma_eval = load_evaluation(FILES["gemma_eval"], "gemma")
    qwen_eval = load_evaluation(FILES["qwen_eval"], "qwen")

    for model, evaluation in [
        ("Gemma", gemma_eval), ("Qwen", qwen_eval)
    ]:
        expected = combined[keys].sort_values(keys).reset_index(drop=True)
        actual = evaluation[keys].sort_values(keys).reset_index(drop=True)

        if not expected.equals(actual):
            raise ValueError(
                f"{model} evaluation rows do not match its generation data."
            )

    combined = combined.merge(
        gemma_eval, on=keys, validate="one_to_one"
    )
    combined = combined.merge(
        qwen_eval, on=keys, validate="one_to_one"
    )

    score_columns = [
        f"{model}_{metric}"
        for model in ["gemma", "qwen"]
        for metric in METRICS
    ]

    for column in score_columns:
        combined[column] = pd.to_numeric(
            combined[column], errors="coerce"
        )

    if combined[score_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric evaluation scores detected."
        )

    if len(combined) != 400:
        raise ValueError(
            f"Expected 400 paired articles; found {len(combined)}."
        )

    # Recalculate lengths identically for both models.
    for column in [
        "human_summary", "gemma_summary", "qwen_summary"
    ]:
        combined[f"{column}_tokens"] = combined[column].map(
            lambda x: len(unicode_tokens(x))
        )
        combined[f"{column}_characters"] = combined[column].map(
            lambda x: len(str(x))
        )
        combined[f"{column}_sentences"] = combined[column].map(
            sentence_count
        )

    for model in ["gemma", "qwen"]:
        combined[f"{model}_length_difference"] = (
            combined[f"{model}_summary_tokens"]
            - combined["human_summary_tokens"]
        )

    combined.to_csv(
        OUT_COMBINED, index=False, encoding="utf-8-sig"
    )

    # ---------- Summary table ----------
    summary_rows = []

    for language, group in combined.groupby("language"):
        for model in ["gemma", "qwen"]:
            row = {
                "language": language,
                "model": model,
                "n": len(group),
            }

            for metric in METRICS:
                values = group[f"{model}_{metric}"]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)

            row["reference_tokens_mean"] = (
                group["human_summary_tokens"].mean()
            )
            row["summary_tokens_mean"] = (
                group[f"{model}_summary_tokens"].mean()
            )
            row["summary_tokens_std"] = (
                group[f"{model}_summary_tokens"].std(ddof=1)
            )
            row["length_difference_mean"] = (
                group[f"{model}_length_difference"].mean()
            )
            row["length_difference_median"] = (
                group[f"{model}_length_difference"].median()
            )

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT_SUMMARY, index=False, encoding="utf-8-sig"
    )

    # ---------- Length table ----------
    length_rows = []

    for language, group in combined.groupby("language"):
        for model in ["gemma", "qwen"]:
            for measure, column in [
                ("reference_tokens", "human_summary_tokens"),
                ("summary_tokens", f"{model}_summary_tokens"),
                ("summary_characters", f"{model}_summary_characters"),
                ("summary_sentences", f"{model}_summary_sentences"),
                ("length_difference_tokens", f"{model}_length_difference"),
            ]:
                values = group[column]
                length_rows.append({
                    "language": language,
                    "model": model,
                    "measure": measure,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                })

    pd.DataFrame(length_rows).to_csv(
        OUT_LENGTH, index=False, encoding="utf-8-sig"
    )

    # ---------- Paired bootstrap confidence intervals ----------
    bootstrap_rows = []

    for language, group in combined.groupby("language"):
        for metric in METRICS:
            n, difference, low, high = paired_bootstrap(
                group[f"gemma_{metric}"],
                group[f"qwen_{metric}"],
                seed=2026,
            )

            bootstrap_rows.append({
                "language": language,
                "metric": metric,
                "comparison": "Qwen minus Gemma",
                "n_paired": n,
                "mean_difference": difference,
                "ci_95_lower": low,
                "ci_95_upper": high,
            })

    pd.DataFrame(bootstrap_rows).to_csv(
        OUT_BOOTSTRAP, index=False, encoding="utf-8-sig"
    )

    # ---------- Comparison plots ----------
    languages = list(combined["language"].drop_duplicates())

    for metric in [
        "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(languages))
        width = 0.34

        for offset, model in [
            (-width / 2, "gemma"), (width / 2, "qwen")
        ]:
            means = [
                combined.loc[
                    combined["language"] == language,
                    f"{model}_{metric}"
                ].mean()
                for language in languages
            ]
            stds = [
                combined.loc[
                    combined["language"] == language,
                    f"{model}_{metric}"
                ].std(ddof=1)
                for language in languages
            ]

            ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=4, label=model.capitalize()
            )

        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylabel("Mean score ± SD")
        ax.set_title(f"{metric.upper()} by language and model")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"{metric}_comparison.png", dpi=200
        )
        plt.close(fig)

    # Summary length plot using the same tokenizer for both models.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(languages))
    width = 0.34

    for offset, model in [
        (-width / 2, "gemma"), (width / 2, "qwen")
    ]:
        means = [
            combined.loc[
                combined["language"] == language,
                f"{model}_summary_tokens"
            ].mean()
            for language in languages
        ]
        stds = [
            combined.loc[
                combined["language"] == language,
                f"{model}_summary_tokens"
            ].std(ddof=1)
            for language in languages
        ]
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4, label=model.capitalize()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel("Unicode-aware token count (mean ± SD)")
    ax.set_title("Generated summary length by language and model")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "summary_length_comparison.png", dpi=200
    )
    plt.close(fig)

    # ---------- Print results ----------
    display_columns = [
        "language", "model", "n",
        "rouge1_f1_mean", "rouge2_f1_mean", "rougeL_f1_mean",
        "bertscore_f1_mean", "summary_tokens_mean",
        "length_difference_mean",
    ]

    print("\nComparison complete.")
    print("Paired articles:", len(combined))
    print("\nMean scores by language and model:")
    print(summary[display_columns].to_string(index=False))

    print("\nSaved files:")
    for path in [
        OUT_COMBINED, OUT_SUMMARY, OUT_LENGTH, OUT_BOOTSTRAP
    ]:
        print(path.resolve())

    print("\nFigures saved in:")
    print(FIGURES.resolve())


if __name__ == "__main__":
    main()
