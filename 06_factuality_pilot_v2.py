from pathlib import Path
import json
import re
import time
import pandas as pd
from openai import OpenAI

# Local, second-pass factuality review. Does NOT overwrite the original pilot.
RESULTS = Path("results")
INPUT_FILE = RESULTS / "gemma_factuality_pilot.csv"
OUTPUT_FILE = RESULTS / "gemma_factuality_pilot_v2.csv"
MODEL_ID = "gemma-3-4b-instruct"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MAX_ARTICLE_CHARS = 10000
MAX_NEW_TOKENS = 900
MAX_ATTEMPTS = 2

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300.0, max_retries=1)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE.resolve()}")
source = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", dtype={"id": str})
source = source[source["judge_status"].eq("success")].copy()

if OUTPUT_FILE.exists():
    out = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig", dtype={"id": str})
else:
    out = pd.DataFrame(columns=[
        "language", "id", "judge_model", "status", "error",
        "article_truncated", "review_json"
    ])

models = [m.id for m in client.models.list().data]
if MODEL_ID not in models:
    raise RuntimeError(f"Model {MODEL_ID!r} not found. Available: {models}")

LABELS = {"supported", "contradicted", "unsupported", "unclear"}
OVERALL = {"faithful", "mostly_faithful", "mixed", "unfaithful", "unclear"}


def parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def validate(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        raise ValueError("Expected a JSON object with a claims list")
    if len(obj["claims"]) > 5:
        raise ValueError("Too many claims; maximum is 5")
    if obj.get("overall_assessment") not in OVERALL:
        raise ValueError("Invalid overall_assessment")
    for i, c in enumerate(obj["claims"]):
        if not isinstance(c, dict):
            raise ValueError(f"Claim {i} is not an object")
        for key in ("claim", "label", "evidence", "reason"):
            if key not in c:
                raise ValueError(f"Claim {i} missing {key}")
        if c["label"] not in LABELS:
            raise ValueError(f"Claim {i} has invalid label {c['label']!r}")
    if not isinstance(obj.get("important_omissions", []), list):
        raise ValueError("important_omissions must be a list")
    return obj


def build_prompt(row, compact=False):
    article = str(row["article"])
    truncated = len(article) > MAX_ARTICLE_CHARS
    article = article[:MAX_ARTICLE_CHARS]
    lang = str(row["language"])
    summary = str(row["gemma_summary"])
    human = str(row.get("human_summary", ""))
    claim_limit = 3 if compact else 5
    return f'''You are a cautious source-grounded fact checker. Evaluate the GENERATED SUMMARY only against the ARTICLE below. Do not use outside knowledge. Language: {lang}.

ARTICLE:
{article}

GENERATED SUMMARY:
{summary}

HUMAN REFERENCE (context only; do not treat it as the source of truth):
{human}

Extract up to {claim_limit} distinct factual claims actually made in the generated summary. Do not invent claims or rewrite them into stronger claims. Split claims when different details could have different truth values. For each claim use one label:
- supported: article explicitly or clearly supports it
- contradicted: article explicitly conflicts with it
- unsupported: article does not establish it, but does not explicitly conflict
- unclear: cannot decide because wording or evidence is ambiguous/incomplete

Evidence must be a short exact quote from the article that supports or contradicts the claim. If no relevant quote exists, use an empty evidence string. A quote must support the specific claim, not merely discuss the same topic. Never mark contradicted only because something is absent. If article truncation might hide evidence, prefer unclear over unsupported.

Return only valid JSON, no Markdown:
{{
  "claims": [{{"claim":"claim from generated summary","label":"supported","evidence":"exact short article quote or empty string","reason":"brief explanation"}}],
  "important_omissions": ["only major source facts omitted by generated summary"],
  "overall_assessment":"mostly_faithful",
  "assessment_reason":"brief reason"
}}
Allowed overall_assessment values: faithful, mostly_faithful, mixed, unfaithful, unclear.
Article truncated for this review: {str(truncated).lower()}
''', truncated


def save(record):
    global out
    mask = (out["language"].astype(str) == str(record["language"])) & (out["id"].astype(str) == str(record["id"]))
    out = out.loc[~mask].copy()
    out = pd.concat([out, pd.DataFrame([record])], ignore_index=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

for _, row in source.iterrows():
    lang, example_id = str(row["language"]), str(row["id"])
    existing = out[(out["language"].astype(str) == lang) & (out["id"].astype(str) == example_id)]
    if not existing.empty and existing.iloc[-1]["status"] == "success":
        print(f"Skipping completed: {lang} | {example_id}")
        continue

    print(f"\nSecond-pass review: {lang} | {example_id}")
    review = None
    error = ""
    truncated = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prompt, truncated = build_prompt(row, compact=(attempt > 1))
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=MAX_NEW_TOKENS,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("Output token limit reached; JSON may be incomplete")
            review = validate(parse_json(choice.message.content))
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"Attempt {attempt} failed: {error}")

    record = {
        "language": lang,
        "id": example_id,
        "judge_model": MODEL_ID,
        "status": "success" if review is not None else "error",
        "error": "" if review is not None else error,
        "article_truncated": truncated,
        "review_json": json.dumps(review, ensure_ascii=False) if review is not None else "",
    }
    save(record)
    if review is not None:
        counts = {}
        for claim in review["claims"]:
            counts[claim["label"]] = counts.get(claim["label"], 0) + 1
        print("Claim labels:", counts, "| overall:", review["overall_assessment"])
    time.sleep(0.2)

print("\nSecond-pass review finished.")
print("Saved:", OUTPUT_FILE.resolve())
print("Status counts:")
print(out["status"].value_counts(dropna=False).to_string())
print("\nReminder: this is still an automated second opinion, not human validation.")
