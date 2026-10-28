from pathlib import Path
import pandas as pd

RESULTS = Path("results")
INPUT_FILE = RESULTS / "gemma_factuality_claims_review.csv"
OUTPUT_FILE = RESULTS / "gemma_factuality_manual_audit.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing file: {INPUT_FILE.resolve()}")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Add blank fields for independent human review.
df["human_label"] = ""
df["evidence_adequate"] = ""
df["human_notes"] = ""

df["label_agreement"] = ""

# Preserve the automated label and evidence unchanged.
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Manual audit sheet created:")
print(OUTPUT_FILE.resolve())
print("\nClaims available:", len(df))
print("\nAutomated label counts:")
print(df["label"].value_counts(dropna=False))

print(
    "\nOpen this CSV in Excel. Review the claim against the original article "
    "and fill in human_label, evidence_adequate, and human_notes."
)
