from pathlib import Path
import json
import pandas as pd

RESULTS = Path("results")
INPUT_FILE = RESULTS / "gemma_factuality_pilot.csv"
CLAIMS_FILE = RESULTS / "gemma_factuality_claims_review.csv"
SUMMARY_FILE = RESULTS / "gemma_factuality_pilot_summary.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE.resolve()}")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", dtype={"id": str})
df = df[df["judge_status"] == "success"].copy()

claim_rows = []
summary_rows = []

for _, row in df.iterrows():
    try:
        judgment = json.loads(row["judge_json"])
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"Could not parse {row['language']} / {row['id']}: {exc}")
        continue

    claims = judgment.get("claims", [])

    for number, claim in enumerate(claims, start=1):
        claim_rows.append({
            "language": row["language"],
            "id": row["id"],
            "claim_number": number,
            "claim": claim.get("claim", ""),
            "label": claim.get("label", "missing"),
            "evidence": claim.get("evidence", ""),
            "reason": claim.get("reason", ""),
            "article_truncated": row.get(
                "article_truncated_for_evaluation", False
            ),
        })

    summary_rows.append({
        "language": row["language"],
        "id": row["id"],
        "number_of_claims": len(claims),
        "supported": sum(c.get("label") == "supported" for c in claims),
        "contradicted": sum(c.get("label") == "contradicted" for c in claims),
        "unsupported": sum(c.get("label") == "unsupported" for c in claims),
        "unclear": sum(c.get("label") == "unclear" for c in claims),
        "overall_assessment": judgment.get("overall_assessment", ""),
        "assessment_reason": judgment.get("assessment_reason", ""),
        "important_omissions": " | ".join(
            judgment.get("important_omissions", [])
        ),
        "article_truncated": row.get(
            "article_truncated_for_evaluation", False
        ),
    })

claims_df = pd.DataFrame(claim_rows)
summary_df = pd.DataFrame(summary_rows)

claims_df.to_csv(CLAIMS_FILE, index=False, encoding="utf-8-sig")
summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

print("Claim-level report:", CLAIMS_FILE.resolve())
print("Summary-level report:", SUMMARY_FILE.resolve())

print("\nClaim labels by language:")
if not claims_df.empty:
    print(pd.crosstab(claims_df["language"], claims_df["label"], margins=True))

print("\nOverall assessments by language:")
if not summary_df.empty:
    print(pd.crosstab(
        summary_df["language"],
        summary_df["overall_assessment"],
        margins=True
    ))

print("\nFirst 10 claims for manual review:")
if not claims_df.empty:
    print(claims_df[[
        "language", "id", "claim", "label", "evidence", "reason"
    ]].head(10).to_string(index=False))
