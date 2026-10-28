
from pathlib import Path
import pandas as pd
import random

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_DIR = Path(
    r"C:\Users\Suryank\Documents\xlsum_gemma_research"
)
RESULTS_DIR = PROJECT_DIR / "results"

GEMMA_FILE = RESULTS_DIR / "gemma_generations_checkpoint.csv"
QWEN_FILE = RESULTS_DIR / "qwen_generations_checkpoint.csv"

OUTPUT_FILE = RESULTS_DIR / "human_eval_blinded.csv"
KEY_FILE = RESULTS_DIR / "human_eval_key.csv"
TEMPLATE_FILE = RESULTS_DIR / "human_eval_template.csv"

SAMPLES_PER_LANGUAGE = 50
RANDOM_SEED = 2026

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_generation_file(path, model_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "language",
        "id",
        "article",
        "human_summary",
        f"{model_name}_summary",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["id"] = df["id"].astype(str).str.strip()

    # Preserve the original article and reference summary.
    keep = [
        "language", "id", "article", "human_summary",
        "title", f"{model_name}_summary"
    ]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()
    df = df.rename(columns={
        f"{model_name}_summary": f"{model_name}_summary"
    })

    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gemma = load_generation_file(GEMMA_FILE, "gemma")
    qwen = load_generation_file(QWEN_FILE, "qwen")

    # Merge only matching article-language pairs.
    merged = gemma.merge(
        qwen,
        on=["language", "id"],
        how="inner",
        suffixes=("_gemma", "_qwen"),
        validate="one_to_one",
    )

    # Choose the best available copy of shared fields.
    for field in ["article", "human_summary", "title"]:
        left = f"{field}_gemma"
        right = f"{field}_qwen"

        if left in merged.columns and right in merged.columns:
            merged[field] = merged[left].fillna(merged[right])
        elif left in merged.columns:
            merged[field] = merged[left]
        elif right in merged.columns:
            merged[field] = merged[right]
        else:
            merged[field] = ""

    merged["gemma_summary"] = merged["gemma_summary"].map(clean_text)
    merged["qwen_summary"] = merged["qwen_summary"].map(clean_text)
    merged["article"] = merged["article"].map(clean_text)
    merged["human_summary"] = merged["human_summary"].map(clean_text)

    # Exclude rows missing any essential text.
    merged = merged[
        (merged["article"] != "")
        & (merged["human_summary"] != "")
        & (merged["gemma_summary"] != "")
        & (merged["qwen_summary"] != "")
    ].copy()

    rng = random.Random(RANDOM_SEED)
    selected_parts = []

    # Accept common dataset language labels.
    language_aliases = {
        "bengali": ["bengali", "bn"],
        "hindi": ["hindi", "hi"],
    }

    for language, aliases in language_aliases.items():
        group = merged[merged["language"].isin(aliases)].copy()

        if len(group) < SAMPLES_PER_LANGUAGE:
            raise ValueError(
                f"Only {len(group)} usable matched {language} articles "
                f"found; need {SAMPLES_PER_LANGUAGE}. "
                "Check language labels and successful generations."
            )

        # Deterministic sample, independent of row order.
        group = group.sample(
            n=SAMPLES_PER_LANGUAGE,
            random_state=RANDOM_SEED,
        )
        selected_parts.append(group)

    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    blinded_rows = []
    key_rows = []

    for _, row in sample.iterrows():
        # Randomize model-to-label mapping separately per article.
        if rng.choice([True, False]):
            summary_a = row["gemma_summary"]
            summary_b = row["qwen_summary"]
            model_a, model_b = "Gemma 3 4B", "Qwen3-4B"
        else:
            summary_a = row["qwen_summary"]
            summary_b = row["gemma_summary"]
            model_a, model_b = "Qwen3-4B", "Gemma 3 4B"

        common = {
            "language": row["language"],
            "article_id": row["id"],
            "title": row.get("title", ""),
            "article": row["article"],
            "reference_summary": row["human_summary"],
            "summary_A": summary_a,
            "summary_B": summary_b,
        }

        blinded_rows.append(common)
        key_rows.append({
            "language": row["language"],
            "article_id": row["id"],
            "model_A": model_a,
            "model_B": model_b,
        })

    blinded = pd.DataFrame(blinded_rows)
    key = pd.DataFrame(key_rows)

    # Annotation fields: use scores independently for each summary.
    annotation_columns = [
        "A_factuality_1_to_5",
        "A_coverage_1_to_5",
        "A_readability_1_to_5",
        "A_unsupported_claim_count",
        "A_factual_error_types",
        "A_omitted_key_facts",
        "A_notes",
        "B_factuality_1_to_5",
        "B_coverage_1_to_5",
        "B_readability_1_to_5",
        "B_unsupported_claim_count",
        "B_factual_error_types",
        "B_omitted_key_facts",
        "B_notes",
        "annotator_id",
    ]

    for column in annotation_columns:
        blinded[column] = ""

    # Template contains the same blinded summaries and empty score fields.
    # Annotate a copy of this file, leaving the original blinded file intact.
    blinded.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    key.to_csv(KEY_FILE, index=False, encoding="utf-8-sig")
    blinded.to_csv(TEMPLATE_FILE, index=False, encoding="utf-8-sig")

    print("\nHuman-evaluation sample created successfully.")
    print(f"Matched articles available: {len(merged)}")
    print(f"Selected articles: {len(blinded)}")
    print(blinded["language"].value_counts().to_string())
    print(f"\nBlinded sample: {OUTPUT_FILE}")
    print(f"Annotation template: {TEMPLATE_FILE}")
    print(f"Model key (keep separate): {KEY_FILE}")
    print("\nNo model inference was performed.")
    print("No generation checkpoint was modified.")


if __name__ == "__main__":
    main()
