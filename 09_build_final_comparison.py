from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Change PROJECT_DIR only if your project is stored elsewhere.
PROJECT_DIR = Path(r"C:\Users\Suryank\Documents\xlsum_gemma_research")
RESULTS = PROJECT_DIR / "results"

AI_FILE = RESULTS / "human_eval_all100_ai_assisted_screening.csv"
KEY_FILE = RESULTS / "human_eval_key.csv"
METRICS_FILE = RESULTS / "qwen_evaluation_per_example.csv"

OUT_SUMMARY = RESULTS / "final_comparison_summary.csv"
OUT_JOINED = RESULTS / "final_comparison_per_article.csv"
OUT_NOTE = RESULTS / "final_comparison_readme.txt"
FIG_DIR = RESULTS / "final_comparison_figures"


def normalize_language(value):
    v = str(value).strip().lower()
    if v in {"hi", "hindi"}:
        return "hindi"
    if v in {"bn", "bengali"}:
        return "bengali"
    return v


def main():
    for path in [AI_FILE, KEY_FILE, METRICS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    ai = pd.read_csv(AI_FILE, encoding="utf-8-sig")
    key = pd.read_csv(KEY_FILE, encoding="utf-8-sig")
    metrics = pd.read_csv(METRICS_FILE, encoding="utf-8-sig")

    # Validate required columns before analysis.
    ai_required = {"language", "article_id", "A_factuality_1_to_5", "A_coverage_1_to_5",
                   "B_factuality_1_to_5", "B_coverage_1_to_5"}
    key_required = {"language", "article_id", "model_A", "model_B"}
    metric_required = {"language", "id", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"}
    for name, df, required in [("AI screening", ai, ai_required), ("model key", key, key_required), ("metrics", metrics, metric_required)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} file is missing columns {sorted(missing)}. Found: {list(df.columns)}")

    ai["language"] = ai["language"].map(normalize_language)
    key["language"] = key["language"].map(normalize_language)
    metrics["language"] = metrics["language"].map(normalize_language)
    ai["article_id"] = ai["article_id"].astype(str).str.strip()
    key["article_id"] = key["article_id"].astype(str).str.strip()
    metrics["id"] = metrics["id"].astype(str).str.strip()

    # Convert A/B screening scores to numeric; blank values remain missing.
    score_cols = ["A_factuality_1_to_5", "A_coverage_1_to_5", "A_readability_1_to_5",
                  "A_unsupported_claim_count", "B_factuality_1_to_5", "B_coverage_1_to_5",
                  "B_readability_1_to_5", "B_unsupported_claim_count"]
    for col in score_cols:
        if col in ai.columns:
            ai[col] = pd.to_numeric(ai[col], errors="coerce")

    # Merge anonymized A/B to model identities only after evaluation is complete.
    ai = ai.merge(key, on=["language", "article_id"], how="left", validate="one_to_one")
    ai["gemma_factuality_preliminary"] = ai.apply(
        lambda r: r.get("A_factuality_1_to_5") if str(r.get("model_A", "")).lower().startswith("gemma") else r.get("B_factuality_1_to_5"), axis=1)
    ai["qwen_factuality_preliminary"] = ai.apply(
        lambda r: r.get("A_factuality_1_to_5") if "qwen" in str(r.get("model_A", "")).lower() else r.get("B_factuality_1_to_5"), axis=1)
    ai["gemma_coverage_preliminary"] = ai.apply(
        lambda r: r.get("A_coverage_1_to_5") if str(r.get("model_A", "")).lower().startswith("gemma") else r.get("B_coverage_1_to_5"), axis=1)
    ai["qwen_coverage_preliminary"] = ai.apply(
        lambda r: r.get("A_coverage_1_to_5") if "qwen" in str(r.get("model_A", "")).lower() else r.get("B_coverage_1_to_5"), axis=1)

    metrics = metrics.rename(columns={"id": "article_id"})
    # Keep relevant metric columns and avoid accidental duplicate content columns.
    metric_cols = [c for c in ["language", "article_id", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_precision", "bertscore_recall", "bertscore_f1"] if c in metrics.columns]
    metrics = metrics[metric_cols].drop_duplicates(["language", "article_id"])

    joined = ai.merge(metrics, on=["language", "article_id"], how="left", validate="one_to_one")
    joined.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")

    rows = []
    for lang, group in joined.groupby("language", dropna=False):
        row = {"language": lang, "n_articles_in_ai_sheet": len(group)}
        for col in ["gemma_factuality_preliminary", "qwen_factuality_preliminary", "gemma_coverage_preliminary", "qwen_coverage_preliminary"]:
            row[col + "_mean"] = group[col].mean()
            row[col + "_n_scored"] = int(group[col].notna().sum())
        for col in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_precision", "bertscore_recall", "bertscore_f1"]:
            if col in group.columns:
                row[col + "_mean"] = group[col].mean()
                row[col + "_n"] = int(group[col].notna().sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Plot preliminary screening means separately from established automatic metrics.
    plot_specs = [
        ("gemma_factuality_preliminary", "qwen_factuality_preliminary", "Preliminary AI-screening factuality (1–5)", "factuality_preliminary.png"),
        ("gemma_coverage_preliminary", "qwen_coverage_preliminary", "Preliminary AI-screening coverage (1–5)", "coverage_preliminary.png"),
    ]
    for left, right, title, filename in plot_specs:
        means = joined.groupby("language")[[left, right]].mean(numeric_only=True)
        means = means.rename(columns={left: "Gemma (preliminary)", right: "Qwen (preliminary)"})
        ax = means.plot(kind="bar", figsize=(8, 5), rot=0)
        ax.set_title(title)
        ax.set_ylabel("Mean score")
        ax.set_xlabel("Language")
        ax.set_ylim(0, 5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=160)
        plt.close()

    note = """FINAL COMPARISON FILES

final_comparison_summary.csv: language-level means and available sample counts.
final_comparison_per_article.csv: merged per-article metrics and preliminary AI-screening fields.
final_comparison_figures/: plots of preliminary factuality and coverage screening scores.

IMPORTANT INTERPRETATION NOTE
The factuality and coverage fields in human_eval_all100_ai_assisted_screening.csv are preliminary AI-assisted screening results, not verified human annotations. Do not describe them as ground truth or use them as definitive evidence that one model is better. Validate a subset using the article text as the only factual evidence. ROUGE/BERTScore values are taken from the existing per-example metrics file and are not recalculated here.

This script reads existing files and creates new outputs only. It does not run model inference or modify generation checkpoints.
"""
    OUT_NOTE.write_text(note, encoding="utf-8")

    print("Comparison outputs created:")
    print(OUT_SUMMARY)
    print(OUT_JOINED)
    print(OUT_NOTE)
    print(FIG_DIR)
    print("\nSummary preview:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
