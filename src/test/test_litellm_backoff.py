from utils.utils import get_env
from agents.models import LiteLLMModelWithBackOff
from agents.prompts import vanilla_prompt

BASE_URL, API_KEY, MODEL = get_env()


# --- Initialize the LiteLLM Model with BackOff ---
# Initialize LiteLLMModelWithBackOff from custom_litellm
litellm_model_backoff = LiteLLMModelWithBackOff(
    model_id=f"litellm_proxy/{MODEL}",
    api_base=BASE_URL,
    api_key=API_KEY,
    temperature=0.7,
    max_tokens=1500
)

print(f"Sending request to model: {MODEL} via LiteLLMModelWithBackOff...")

try:
    # Send message using LiteLLMModelWithBackOff with proper message format
    messages = [{"role": "user", "content": vanilla_prompt}]
    response = litellm_model_backoff(messages)

    print("\n--- Response ---")
    print(response)

    print("\n--- Test completed successfully ---")
    print(f"Model used: {MODEL}")
    print(f"Max tokens configured: {litellm_model_backoff.max_tokens}")

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
