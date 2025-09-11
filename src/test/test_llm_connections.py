"""
Consolidated LLM connection tests.
Combines functionality from test_litellm.py, test_litellm_backoff.py, and test_llm_connection.py
"""

from openai import OpenAI
from smolagents import LiteLLMModel
from utils.utils import get_env
from agents.models import LiteLLMModelWithBackOff
from agents.prompts import vanilla_prompt


def test_openai_client_connection():
    """Test direct OpenAI client connection."""
    print("\n=== Testing OpenAI Client Connection ===")
    
    config = get_env()
    BASE_URL = config["BASE_URL"]
    API_KEY = config["API_KEY"]
    MODEL = config["MODEL"]
    
    # Initialize the OpenAI Client pointing to our platform
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
        
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def test_smolagents_litellm_model():
    """Test smolagents LiteLLMModel connection."""
    print("\n=== Testing Smolagents LiteLLMModel ===")
    
    config = get_env()
    BASE_URL = config["BASE_URL"]
    API_KEY = config["API_KEY"]
    MODEL = config["MODEL"]

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
        
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def test_litellm_model_with_backoff():
    """Test LiteLLMModelWithBackOff connection."""
    print("\n=== Testing LiteLLMModelWithBackOff ===")
    
    config = get_env()
    BASE_URL = config["BASE_URL"]
    API_KEY = config["API_KEY"]
    MODEL = config["MODEL"]

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
        
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


def run_all_connection_tests():
    """Run all LLM connection tests."""
    print("🚀 Running All LLM Connection Tests")
    print("=" * 50)
    
    results = {
        "OpenAI Client": test_openai_client_connection(),
        "Smolagents LiteLLM": test_smolagents_litellm_model(),
        "LiteLLM with BackOff": test_litellm_model_with_backoff()
    }
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All connection tests passed!")
    else:
        print("\n⚠️  Some connection tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    run_all_connection_tests()
