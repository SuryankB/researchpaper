from pathlib import Path
import json
import re
import time
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS = Path("results")
INPUT_FILE = RESULTS / "gemma_generations_checkpoint.csv"
OUTPUT_FILE = RESULTS / "gemma_factuality_pilot.csv"

MODEL_ID = "gemma-3-4b-instruct"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

EXAMPLES_PER_LANGUAGE = 10
RANDOM_SEED = 42
MAX_NEW_TOKENS = 1200
MAX_ARTICLE_CHARS = 10000
MAX_RETRIES = 2

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=300.0,
    max_retries=1,
)

# ============================================================
# LOAD INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    dtype={"id": str},
)

df = df[df["status"] == "success"].copy()

required_columns = {
    "language",
    "id",
    "article",
    "human_summary",
    "gemma_summary",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

for column in ["article", "human_summary", "gemma_summary"]:
    df[column] = df[column].fillna("").astype(str).str.strip()

df = df[
    df["article"].ne("")
    & df["human_summary"].ne("")
    & df["gemma_summary"].ne("")
].copy()

# Select the same reproducible 10 examples per language.
pilot_parts = []

for language in ["Hindi", "Bengali"]:
    language_df = df[df["language"] == language].copy()

    if len(language_df) < EXAMPLES_PER_LANGUAGE:
        raise ValueError(
            f"Only {len(language_df)} examples available for {language}."
        )

    selected = language_df.sample(
        n=EXAMPLES_PER_LANGUAGE,
        random_state=RANDOM_SEED,
    )
    pilot_parts.append(selected)

pilot = pd.concat(pilot_parts, ignore_index=True)

# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

output_columns = [
    "language",
    "id",
    "article",
    "human_summary",
    "gemma_summary",
    "article_truncated_for_evaluation",
    "judge_model",
    "judge_status",
    "judge_error",
    "judge_json",
]

if OUTPUT_FILE.exists():
    results = pd.read_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        dtype={"id": str},
    )

    for column in output_columns:
        if column not in results.columns:
            results[column] = ""

    results = results[output_columns].copy()
else:
    results = pd.DataFrame(columns=output_columns)


def save_result(record):
    """Replace the existing record for this language/ID and save."""
    global results

    same_record = (
        (results["language"].astype(str) == str(record["language"]))
        & (results["id"].astype(str) == str(record["id"]))
    )

    results = results.loc[~same_record].copy()
    results = pd.concat(
        [results, pd.DataFrame([record])],
        ignore_index=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MODEL CHECK
# ============================================================

available_models = [
    model.id for model in client.models.list().data
]

print("Available models:", available_models)

if MODEL_ID not in available_models:
    raise RuntimeError(
        f"Model '{MODEL_ID}' is not available. "
        f"Available models: {available_models}"
    )


# ============================================================
# PROMPT AND JSON VALIDATION
# ============================================================

ALLOWED_LABELS = {
    "supported",
    "contradicted",
    "unsupported",
    "unclear",
}

ALLOWED_ASSESSMENTS = {
    "faithful",
    "mostly_faithful",
    "mixed",
    "unfaithful",
    "unclear",
}


def make_prompt(language, article, human_summary, gemma_summary):
    article_truncated = len(article) > MAX_ARTICLE_CHARS
    article_for_eval = article[:MAX_ARTICLE_CHARS]

    return f"""
You are a cautious factuality evaluator for news summaries.

Evaluate the GEMMA SUMMARY using only the ARTICLE below.
The text is in {language}. Preserve its meaning when evaluating claims.
Do not use outside knowledge.

ARTICLE:
{article_for_eval}

HUMAN REFERENCE SUMMARY (context only):
{human_summary}

GEMMA SUMMARY:
{gemma_summary}

Instructions:
- Evaluate factual claims in the Gemma summary.
- Use no more than 5 claims. Combine closely related details when sensible.
- Each claim must receive exactly one label:
  supported, contradicted, unsupported, or unclear.
- supported: the article provides evidence for the claim.
- contradicted: the article provides evidence that conflicts with the claim.
- unsupported: the supplied article does not establish the claim.
- unclear: the evidence is ambiguous or incomplete.
- Do not confuse unsupported with contradicted.
- Use a short exact article quotation as evidence when possible.
- If unsupported, explain what evidence is missing.
- List only important omissions, not every detail excluded by summarization.
- Do not assume the human reference is complete or always correct.
- The article may be cut off at the end. Be conservative if evidence may be missing.
- Keep explanations short.
- Return valid JSON only. No Markdown fences and no text outside the JSON.

Use exactly this structure:
{{
  "claims": [
    {{
      "claim": "short factual claim",
      "label": "supported",
      "evidence": "short exact quote from the article or empty string",
      "reason": "short explanation"
    }}
  ],
  "important_omissions": [
    "short description of an important omitted fact"
  ],
  "overall_assessment": "faithful",
  "assessment_reason": "short explanation"
}}

Allowed overall_assessment values:
faithful, mostly_faithful, mixed, unfaithful, unclear.

Article was truncated for this evaluation: {str(article_truncated).lower()}
"""


def parse_json_response(text):
    """Parse JSON, including common Markdown wrappers."""
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    # First try to parse the entire response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recover a JSON object if the model added surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("No complete JSON object found in model response.")


def validate_response(parsed):
    """Reject incomplete or invalid structures."""
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object.")

    claims = parsed.get("claims")
    omissions = parsed.get("important_omissions")
    assessment = parsed.get("overall_assessment")

    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list.")

    if len(claims) > 5:
        raise ValueError("More than 5 claims returned.")

    if not isinstance(omissions, list):
        raise ValueError("'important_omissions' must be a list.")

    if assessment not in ALLOWED_ASSESSMENTS:
        raise ValueError(
            f"Invalid overall assessment: {assessment!r}"
        )

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"Claim {index} must be an object.")

        for field in ["claim", "label", "evidence", "reason"]:
            if field not in claim:
                raise ValueError(
                    f"Claim {index} missing field: {field}"
                )

        if claim["label"] not in ALLOWED_LABELS:
            raise ValueError(
                f"Invalid label in claim {index}: {claim['label']!r}"
            )

    if any(not isinstance(item, str) for item in omissions):
        raise ValueError("Every omission must be a string.")

    return parsed


# ============================================================
# EVALUATE PILOT
# ============================================================

for _, row in pilot.iterrows():
    language = str(row["language"])
    example_id = str(row["id"])

    existing = results[
        (results["language"].astype(str) == language)
        & (results["id"].astype(str) == example_id)
    ]

    # Keep valid previous judgments; retry only failed records.
    if (
        not existing.empty
        and existing.iloc[-1]["judge_status"] == "success"
    ):
        print(f"Already evaluated: {language} | {example_id}")
        continue

    article = str(row["article"])
    human_summary = str(row["human_summary"])
    gemma_summary = str(row["gemma_summary"])

    truncated = len(article) > MAX_ARTICLE_CHARS

    print(f"\nEvaluating {language} | {example_id}")

    record = {
        "language": language,
        "id": example_id,
        "article": article,
        "human_summary": human_summary,
        "gemma_summary": gemma_summary,
        "article_truncated_for_evaluation": truncated,
        "judge_model": MODEL_ID,
        "judge_status": "error",
        "judge_error": "",
        "judge_json": "",
    }

    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = make_prompt(
                language,
                article,
                human_summary,
                gemma_summary,
            )

            if attempt > 1:
                prompt += """
IMPORTANT: Your previous response was not valid JSON.
Return a shorter response with at most 3 claims.
Use very short evidence quotes and explanations.
Return one complete JSON object only.
Do not stop before closing all JSON brackets.
"""

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=MAX_NEW_TOKENS,
            )

            answer = (
                response.choices[0].message.content or ""
            ).strip()

            finish_reason = response.choices[0].finish_reason

            if finish_reason == "length":
                raise ValueError(
                    "Response hit the output-token limit; JSON may be truncated."
                )

            parsed = parse_json_response(answer)
            parsed = validate_response(parsed)

            record["judge_status"] = "success"
            record["judge_error"] = ""
            record["judge_json"] = json.dumps(
                parsed,
                ensure_ascii=False,
            )

            print("Claims evaluated:", len(parsed["claims"]))
            print(
                "Overall assessment:",
                parsed["overall_assessment"],
            )

            break

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: {last_error}"
            )

    if record["judge_status"] != "success":
        record["judge_error"] = last_error
        print("Could not evaluate this example.")

    save_result(record)
    time.sleep(0.3)


# ============================================================
# SUMMARY
# ============================================================

# Restrict summary statistics to this pilot's selected examples.
pilot_keys = set(
    zip(pilot["language"].astype(str), pilot["id"].astype(str))
)

pilot_results = results[
    results.apply(
        lambda r: (str(r["language"]), str(r["id"])) in pilot_keys,
        axis=1,
    )
].copy()

print("\nPilot evaluation finished.")
print("Output file:", OUTPUT_FILE.resolve())

print("\nStatus counts:")
print(pilot_results["judge_status"].value_counts(dropna=False))

print("\nResults by language:")
print(
    pd.crosstab(
        pilot_results["language"],
        pilot_results["judge_status"],
    )
)

print("\nSuccessful judgments:", int(
    (pilot_results["judge_status"] == "success").sum()
))
print("Failed judgments:", int(
    (pilot_results["judge_status"] != "success").sum()
))

print(
    "\nReminder: automated factuality labels are provisional. "
    "Review the article evidence manually before reporting factuality rates."
)
