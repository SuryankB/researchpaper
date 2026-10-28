
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

print("Checking connection to LM Studio...")

models = client.models.list()

if not models.data:
    raise RuntimeError("No models were returned by LM Studio.")

for model in models.data:
    print("Available model:", model.id)

model_id = models.data[0].id

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a news summarization assistant. "
                "Summarize in the same language as the article. "
                "Do not add unsupported facts."
            )
        },
        {
            "role": "user",
            "content": (
                "বাংলাদেশে আজ থেকে গণটিকাদান কর্মসূচি শুরু হচ্ছে। "
                "দুই বাক্যে সংবাদটি সংক্ষেপ করুন।"
            )
        }
    ],
    temperature=0.2,
    max_tokens=150
)

print("\nGemma's Bengali summary:")
print(response.choices[0].message.content)
