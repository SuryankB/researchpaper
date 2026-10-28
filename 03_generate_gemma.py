from pathlib import Path
import time
import pandas as pd
from openai import OpenAI

# ---------- Configuration ----------
RESULTS = Path("results")
SAMPLE_FILE = RESULTS / "xlsum_sample_200_per_language.csv"
CHECKPOINT_FILE = RESULTS / "gemma_generations_checkpoint.csv"

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
MODEL_ID = "gemma-3-4b-instruct"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
SAVE_EVERY = 1  # Save after every completed example

# Start with a small test; increase to 400 after checking results.
TEST_LIMIT = 400

# ---------- Check input ----------
if not SAMPLE_FILE.exists():
    raise FileNotFoundError(
        f"Sample file not found: {SAMPLE_FILE.resolve()}\n"
        "Run 02_make_sample.py first."
    )

df = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig", dtype={"id": str})

required = {"language", "id", "article", "human_summary"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Keep only examples with non-empty articles and references.
df = df.dropna(subset=["article", "human_summary"]).copy()
df["article"] = df["article"].astype(str)
df["human_summary"] = df["human_summary"].astype(str)
df = df[
    df["article"].str.strip().ne("")
    & df["human_summary"].str.strip().ne("")
].copy()

# ---------- Connect to LM Studio ----------
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

available_models = client.models.list().data
model_ids = [model.id for model in available_models]
print("Models available in LM Studio:", model_ids)

if MODEL_ID not in model_ids:
    raise RuntimeError(
        f"Model '{MODEL_ID}' was not found in LM Studio. "
        f"Available models: {model_ids}\n"
        "Load the Gemma model in LM Studio and update MODEL_ID if needed."
    )

# ---------- Load completed generations ----------
if CHECKPOINT_FILE.exists():
    checkpoint = pd.read_csv(
        CHECKPOINT_FILE, encoding="utf-8-sig", dtype={"id": str}
    )
    if not {"language", "id", "gemma_summary", "status"}.issubset(
        checkpoint.columns
    ):
        raise ValueError(
            "Checkpoint file has an unexpected format. "
            "Back it up before changing or deleting it."
        )
else:
    checkpoint = pd.DataFrame(
        columns=[
            "language", "id", "title", "url", "article",
            "human_summary", "gemma_summary", "status", "error"
        ]
    )

completed_keys = set(
    zip(
        checkpoint.loc[checkpoint["status"] == "success", "language"],
        checkpoint.loc[checkpoint["status"] == "success", "id"].astype(str),
    )
)

# TEST_LIMIT=4 tests four examples total, not four per language.
pending = df[
    ~df.apply(
        lambda row: (row["language"], str(row["id"])) in completed_keys,
        axis=1,
    )
].head(TEST_LIMIT)

if pending.empty:
    print("No pending examples in this run.")
    print("If all 4 test examples succeeded, set TEST_LIMIT = 400 to continue.")
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
    new_row = pd.DataFrame([record])

    # Replace an existing row for this language/ID, if any.
    if not checkpoint.empty:
        keep = ~(
            (checkpoint["language"] == record["language"])
            & (checkpoint["id"].astype(str) == str(record["id"]))
        )
        checkpoint = checkpoint.loc[keep].copy()

    checkpoint = pd.concat([checkpoint, new_row], ignore_index=True)
    checkpoint.to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

# ---------- Generate summaries ----------
print(f"\nExamples queued for this run: {len(pending)}")
success_count = 0
failure_count = 0

for position, (_, row) in enumerate(pending.iterrows(), start=1):
    language = str(row["language"])
    example_id = str(row["id"])
    print(f"\n[{position}/{len(pending)}] {language} | {example_id}")

    record = {
        "language": language,
        "id": example_id,
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "article": str(row["article"]),
        "human_summary": str(row["human_summary"]),
        "gemma_summary": "",
        "status": "error",
        "error": "",
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": make_prompt(language, record["article"]),
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        )

        summary = (response.choices[0].message.content or "").strip()
        if not summary:
            raise ValueError("LM Studio returned an empty summary.")

        record["gemma_summary"] = summary
        record["status"] = "success"
        success_count += 1

        print("Human reference:", record["human_summary"][:250])
        print("Gemma summary:", summary[:500])

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        failure_count += 1
        print("ERROR:", record["error"])

    save_checkpoint(record)
    time.sleep(0.2)

print("\nTest run finished.")
print("Successful generations this run:", success_count)
print("Failed generations this run:", failure_count)
print("Checkpoint saved to:", CHECKPOINT_FILE.resolve())
print("Successful generations total:", int((checkpoint["status"] == "success").sum()))
