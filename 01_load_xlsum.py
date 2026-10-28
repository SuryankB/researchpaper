
from pathlib import Path
import pandas as pd

OUT = Path("results")
OUT.mkdir(exist_ok=True)

# Pin the dataset revisions for reproducibility.
FILES = {
    "Hindi": (
        "https://huggingface.co/datasets/csebuetnlp/xlsum/"
        "resolve/651dcf16e996d411b12f0bb7b8fc671788f24193/"
        "hindi/xlsum-test.parquet"
    ),
    "Bengali": (
        "https://huggingface.co/datasets/csebuetnlp/xlsum/"
        "resolve/014d1559dcde03c72beff7d2e617f9792a54fd39/"
        "bengali/xlsum-test.parquet"
    ),
}

all_rows = []

for language, url in FILES.items():
    print(f"\nLoading XLSum {language} test data...")

    # pandas downloads and reads the published Parquet file.
    df = pd.read_parquet(url, engine="pyarrow")

    print("Columns:", df.columns.tolist())
    print("Number of rows:", len(df))
    print("\nFirst row:")
    print(df.iloc[0].to_dict())

    # XLSum Parquet commonly uses id, url, title, text, summary.
    required = {"text", "summary"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns for {language}: {df.columns.tolist()}"
        )

    df["language"] = language

    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    # Save the original test split with article and human summary.
    language_path = OUT / f"xlsum_{language.lower()}_test.csv"
    df.to_csv(language_path, index=False, encoding="utf-8-sig")

    for _, row in df.iterrows():
        all_rows.append({
            "language": language,
            "id": str(row["id"]),
            "article": str(row["text"]),
            "human_summary": str(row["summary"]),
            "title": str(row["title"]) if "title" in df.columns else "",
            "url": str(row["url"]) if "url" in df.columns else "",
        })

master = pd.DataFrame(all_rows)

master_path = OUT / "xlsum_hindi_bengali_test.csv"
master.to_csv(master_path, index=False, encoding="utf-8-sig")

print("\nDataset loading complete.")
print("\nExamples per language:")
print(master.groupby("language").size())

print("\nMissing article counts:")
print(master["article"].astype(str).str.strip().eq("").groupby(
    master["language"]
).sum())

print("\nMissing human summary counts:")
print(master["human_summary"].astype(str).str.strip().eq("").groupby(
    master["language"]
).sum())

print("\nSaved master CSV:", master_path.resolve())
print("Saved language-specific CSVs in:", OUT.resolve())
