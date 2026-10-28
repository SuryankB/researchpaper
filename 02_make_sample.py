from pathlib import Path
import pandas as pd

# Project output folder
RESULTS = Path("results")
INPUT_FILE = RESULTS / "xlsum_hindi_bengali_test.csv"
OUTPUT_FILE = RESULTS / "xlsum_sample_200_per_language.csv"

SAMPLE_SIZE = 200
RANDOM_SEED = 42

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE.resolve()}\n"
        "Run 01_load_xlsum.py first."
    )

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", dtype={"id": str})

required_columns = {
    "language", "id", "article", "human_summary"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

samples = []

for language in ["Hindi", "Bengali"]:
    language_df = df[df["language"] == language].copy()

    if len(language_df) < SAMPLE_SIZE:
        raise ValueError(
            f"{language} has only {len(language_df)} rows; "
            f"{SAMPLE_SIZE} are required."
        )

    # Fixed seed makes the selection reproducible.
    selected = language_df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    )
    samples.append(selected)

sample = pd.concat(samples, ignore_index=True)

# Check uniqueness and completeness.
if sample.duplicated(subset=["language", "id"]).any():
    raise ValueError("Duplicate language/ID pairs found in the sample.")

if sample[["article", "human_summary"]].isna().any().any():
    raise ValueError("Missing article or human summary in the sample.")

sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Sampling complete.")
print(f"Random seed: {RANDOM_SEED}")
print(f"Total examples: {len(sample)}")
print("\nExamples per language:")
print(sample.groupby("language").size())
print(f"\nSaved sample to: {OUTPUT_FILE.resolve()}")

print("\nSample preview:")
print(sample[["language", "id", "title"]].head(6).to_string(index=False))
