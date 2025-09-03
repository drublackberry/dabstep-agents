from smolagents import LiteLLMModel
from utils.utils import get_env
from agents.prompts import vanilla_prompt

BASE_URL, API_KEY, MODEL = get_env()


# --- Initialize the LiteLLM Model ---
# Initialize LiteLLMModel from smolagents
litellm_model = LiteLLMModel(
    model_id=f"litellm_proxy/{MODEL}",
    api_base=BASE_URL,
    api_key=API_KEY,
    temperature=0.7
)

print(f"Sending request to model: {MODEL} via smolagents LiteLLMModel...")

try:
    # Send message using smolagents LiteLLMModel with proper message format
    messages = [{"role": "user", "content": vanilla_prompt}]
    response = litellm_model(messages)

    print("\n--- Response ---")
    print(response)

except Exception as e:
    print(f"An error occurred: {e}")