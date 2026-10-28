
from pathlib import Path
import time
import pandas as pd
from openai import OpenAI

# ---------- Configuration ----------
RESULTS = Path("results")

SAMPLE_FILE = RESULTS / "xlsum_sample_200_per_language.csv"

# NEW FILE: protects your original Gemma results
CHECKPOINT_FILE = RESULTS / "gemma_generations_with_finish_reason.csv"

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL_ID = "gemma-3-4b-instruct"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
SAVE_EVERY = 1

# Process all pending examples (200 Hindi + 200 Bengali).
TEST_LIMIT = 400

# ---------- Check input ----------
if not SAMPLE_FILE.exists():
    raise FileNotFoundError(
        f"Sample file not found: {SAMPLE_FILE.resolve()}\n"
        "Run 02_make_sample.py first."
    )

df = pd.read_csv(
    SAMPLE_FILE,
    encoding="utf-8-sig",
    dtype={"id": str},
)

required = {"language", "id", "article", "human_summary"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Keep examples with non-empty articles and references.
df = df.dropna(subset=["article", "human_summary"]).copy()
df["article"] = df["article"].astype(str)
df["human_summary"] = df["human_summary"].astype(str)
df["id"] = df["id"].astype(str)

df = df[
    df["article"].str.strip().ne("")
    & df["human_summary"].str.strip().ne("")
].copy()

# Avoid duplicate article IDs within a language.
df = df.drop_duplicates(subset=["language", "id"], keep="first")

# ---------- Connect to LM Studio ----------
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

available_models = client.models.list().data
model_ids = [model.id for model in available_models]

print("Models available in LM Studio:", model_ids)

if MODEL_ID not in model_ids:
    raise RuntimeError(
        f"Model '{MODEL_ID}' was not found in LM Studio.\n"
        f"Available models: {model_ids}\n"
        "Load the correct Gemma model or update MODEL_ID."
    )

# ---------- Define checkpoint columns ----------
COLUMNS = [
    "language",
    "id",
    "title",
    "url",
    "article",
    "human_summary",
    "gemma_summary",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "status",
    "error",
]

# Resume only from this NEW checkpoint file.
# Existing original Gemma results are not imported because
# their finish reasons were not saved.
if CHECKPOINT_FILE.exists():
    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig",
        dtype={"id": str},
    )

    for column in COLUMNS:
        if column not in checkpoint.columns:
            checkpoint[column] = ""

    checkpoint = checkpoint[COLUMNS].copy()
    checkpoint["id"] = checkpoint["id"].astype(str)

    # Keep the latest row for each language/article ID.
    checkpoint = checkpoint.drop_duplicates(
        subset=["language", "id"],
        keep="last",
    ).reset_index(drop=True)

else:
    checkpoint = pd.DataFrame(columns=COLUMNS)

def make_key(language, example_id):
    return (str(language), str(example_id))

# Skip successful examples already saved in the NEW checkpoint.
completed_keys = set(
    make_key(row["language"], row["id"])
    for _, row in checkpoint.iterrows()
    if str(row["status"]) == "success"
)

pending = df[
    ~df.apply(
        lambda row: make_key(row["language"], row["id"])
        in completed_keys,
        axis=1,
    )
].head(TEST_LIMIT)

if pending.empty:
    print("No pending examples.")
    print("Checkpoint:", CHECKPOINT_FILE.resolve())
    raise SystemExit(0)

def make_prompt(language, article):
    language_name = {
        "Hindi": "Hindi (हिन्दी)",
        "Bengali": "Bengali (বাংলা)",
    }.get(language, language)

    return f"""You are a careful news summarization assistant.
Write a concise summary in {language_name}, the same language as the article.
Use 1–3 sentences.
Include only information supported by the article.
Do not add outside facts, opinions, or speculation.
Preserve important names, numbers, events, and attribution.
Return only the summary, without a heading or explanation.

ARTICLE:
{article}

SUMMARY:"""

def save_checkpoint(record):
    global checkpoint

    # Replace the row for this article if it already exists.
    if not checkpoint.empty:
        keep = ~(
            (checkpoint["language"].astype(str) == str(record["language"]))
            & (checkpoint["id"].astype(str) == str(record["id"]))
        )
        checkpoint = checkpoint.loc[keep].copy()

    checkpoint = pd.concat(
        [checkpoint, pd.DataFrame([record], columns=COLUMNS)],
        ignore_index=True,
    )

    checkpoint.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

# ---------- Generate summaries ----------
print(f"\nPending examples: {len(pending)}")
print(f"Checkpoint: {CHECKPOINT_FILE.resolve()}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Temperature: {TEMPERATURE}")

success_count = 0
failure_count = 0

for position, (_, row) in enumerate(pending.iterrows(), start=1):
    language = str(row["language"])
    example_id = str(row["id"])

    print(
        f"\n[{position}/{len(pending)}] "
        f"{language} | {example_id}"
    )

    record = {
        "language": language,
        "id": example_id,
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "article": str(row["article"]),
        "human_summary": str(row["human_summary"]),
        "gemma_summary": "",
        "finish_reason": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "status": "error",
        "error": "",
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": make_prompt(
                        language,
                        record["article"],
                    ),
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        )

        choice = response.choices[0]

        record["gemma_summary"] = (
            choice.message.content or ""
        ).strip()

        # Capture why the API says generation ended.
        record["finish_reason"] = choice.finish_reason or ""

        # Token usage is recorded if the server provides it.
        if response.usage is not None:
            record["prompt_tokens"] = (
                response.usage.prompt_tokens
            )
            record["completion_tokens"] = (
                response.usage.completion_tokens
            )

        if not record["gemma_summary"]:
            raise ValueError(
                "LM Studio returned an empty summary."
            )

        record["status"] = "success"
        success_count += 1

        print("Finish reason:", record["finish_reason"])
        print("Completion tokens:", record["completion_tokens"])
        print("Gemma summary:", record["gemma_summary"][:300])

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        failure_count += 1
        print("ERROR:", record["error"])

    # Save after every example so progress is preserved.
    save_checkpoint(record)

    time.sleep(0.2)

# ---------- Final report ----------
print("\nGeneration run finished.")
print("Successful generations this run:", success_count)
print("Failed generations this run:", failure_count)
print("Checkpoint saved to:", CHECKPOINT_FILE.resolve())

print("\nFinish-reason counts for successful saved outputs:")
successful = checkpoint[
    checkpoint["status"].astype(str) == "success"
]

print(
    successful.groupby(
        ["language", "finish_reason"],
        dropna=False,
    ).size().to_string()
)

print("\nTotal successful saved outputs:", len(successful))
