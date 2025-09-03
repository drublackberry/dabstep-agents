from openai import OpenAI
from utils.utils import get_env
from agents.prompts import vanilla_prompt

BASE_URL, API_KEY, MODEL = get_env()

# --- Initialize the OpenAI Client pointing to our platform
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

print(f"Sending request to model: {MODEL}...")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": vanilla_prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    print("\n--- Response ---")
    print(response.choices[0].message.content)
    print("\n--- Full response ---")
    print(response.model_dump_json(indent=2))


except Exception as e:
    print(f"An error occurred: {e}")
