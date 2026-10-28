
import os
import time
import pandas as pd
from openai import OpenAI

# --------------------------------------------------
# Qwen XLSum Hindi/Bengali summarization experiment
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "gemma_generations_checkpoint.csv"
)
OUTPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "qwen_generations_checkpoint.csv"
)

MODEL_ID = "qwen/qwen3-4b-2507"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
SAVE_EVERY = 5

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=300.0,
)

PROMPT_TEMPLATE = """You are a careful news summarization assistant.
Write a concise summary in {language_name}, the same language as the article.
Use 1–3 sentences.
Include only information supported by the article.
Do not add outside facts, opinions, or speculation.
Preserve important names, numbers, events, and attribution.
Return only the summary, without a heading or explanation.

Article:
{article}"""


def generate_summary(language, article):
    prompt = PROMPT_TEMPLATE.format(
        language_name=language,
        article=article,
    )

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    choice = response.choices[0]
    summary = choice.message.content or ""

    return summary.strip(), choice.finish_reason


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    source = pd.read_csv(INPUT_CSV).fillna("")
    if len(source) != 400:
        raise ValueError(
            f"Expected 400 source rows, found {len(source)}. "
            "Check the Gemma checkpoint before continuing."
        )

    # Keep the original article IDs, languages, and references.
    if os.path.exists(OUTPUT_CSV):
        results = pd.read_csv(OUTPUT_CSV).fillna("").to_dict("records")
        existing = {str(row["id"]): row for row in results}
        print(f"Resuming existing Qwen checkpoint: {len(existing)} rows")
    else:
        existing = {}

    for index, row in source.iterrows():
        article_id = str(row["id"])

        # Skip rows already generated successfully.
        if (
            article_id in existing
            and existing[article_id].get("status") == "success"
            and existing[article_id].get("qwen_summary", "").strip()
        ):
            continue

        language = str(row["language"])
        article = str(row["article"])

        if not article.strip():
            existing[article_id] = {
                **row.to_dict(),
                "qwen_summary": "",
                "status": "error",
                "error": "Empty article",
                "finish_reason": "",
            }
            continue

        try:
            summary, finish_reason = generate_summary(language, article)

            if not summary:
                raise ValueError("Model returned an empty summary")

            existing[article_id] = {
                **row.to_dict(),
                "qwen_summary": summary,
                "status": "success",
                "error": "",
                "finish_reason": finish_reason or "",
            }

            print(
                f"[{index + 1}/400] {language} | "
                f"{article_id} | {len(summary)} characters"
            )

        except Exception as exc:
            existing[article_id] = {
                **row.to_dict(),
                "qwen_summary": "",
                "status": "error",
                "error": str(exc),
                "finish_reason": "",
            }
            print(f"[{index + 1}/400] ERROR: {article_id}: {exc}")

        # Save progress regularly, including failures for later retries.
        if (index + 1) % SAVE_EVERY == 0:
            pd.DataFrame(existing.values()).to_csv(
                OUTPUT_CSV, index=False, encoding="utf-8-sig"
            )
            print(f"Checkpoint saved: {OUTPUT_CSV}")

        time.sleep(0.1)

    final = pd.DataFrame(existing.values())

    # Restore the exact source order.
    order = {str(row["id"]): i for i, row in source.iterrows()}
    final["_order"] = final["id"].astype(str).map(order)
    final = final.sort_values("_order").drop(columns="_order")
    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nGeneration finished.")
    print("Total rows:", len(final))
    print("Successful:", (final["status"] == "success").sum())
    print("Errors:", (final["status"] != "success").sum())
    print("Output:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
