from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, AIMessage
from test_utils import TEST_PROMPT, get_env


# Load environment variables from .env file
load_dotenv()
BASE_URL, API_KEY, MODEL = get_env()


# --- Initialize the LiteLLM Client ---
# Initialize ChatLiteLLM
chatllm = ChatLiteLLM(
    model=MODEL,
    api_base=BASE_URL,
    custom_llm_provider="litellm_proxy",
    api_key=API_KEY,
    temperature=0.7
)

print(f"Sending request to model: {MODEL} via litellm client...")

try:
    # Single message
    messages = [HumanMessage(content=TEST_PROMPT)]
    response = chatllm.invoke(messages)

    print("\n--- Response ---")
    print(response.content)

except Exception as e:
    print(f"An error occurred: {e}")