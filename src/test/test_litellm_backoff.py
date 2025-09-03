from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
from custom_litellm import LiteLLMModelWithBackOff
from test_utils import TEST_PROMPT, get_env


# Load environment variables from .env file
load_dotenv()
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
    messages = [{"role": "user", "content": TEST_PROMPT}]
    response = litellm_model_backoff(messages)

    print("\n--- Response ---")
    print(response)

    print("\n--- Test completed successfully ---")
    print(f"Model used: {MODEL}")
    print(f"Max tokens configured: {litellm_model_backoff.max_tokens}")

except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
