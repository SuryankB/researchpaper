from pathlib import Path
import pandas as pd

path = Path("results/gemma_generations_checkpoint.csv")

if not path.exists():
    raise FileNotFoundError(
        "Gemma checkpoint not found. Run the generation script first."
    )

df = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})

print("Total checkpoint rows:", len(df))
print("\nStatus counts:")
print(df["status"].value_counts(dropna=False))

print("\nResults by language and status:")
print(pd.crosstab(df["language"], df["status"], margins=True))

success = df[df["status"] == "success"].copy()

print("\nSuccessful summaries by language:")
print(success.groupby("language").size())

print("\nDuplicate language/ID pairs:")
print(df.duplicated(subset=["language", "id"]).sum())

print("\nEmpty successful summaries:")
print(
    success["gemma_summary"].fillna("").str.strip().eq("").sum()
)

print("\nFailed examples:")
failed = df[df["status"] != "success"]
if failed.empty:
    print("None")
else:
    print(failed[["language", "id", "error"]].to_string(index=False))

print("\nCheckpoint file:", path.resolve())
