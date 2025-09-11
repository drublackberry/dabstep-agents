"""
Pytest test suite for LiteLLM model testing.
Encapsulates logic from test_litellm.py and test_litellm_backoff.py
"""

import pytest
from smolagents import LiteLLMModel
from agents.models import LiteLLMModelWithBackOff
from agents.prompts import vanilla_prompt
from utils.utils import get_env


class TestLiteLLMModels:
    """Test suite for LiteLLM model implementations."""
    
    @pytest.fixture(scope="class")
    def env_config(self):
        """Get environment configuration for testing."""
        try:
            config = get_env()
            return {
                "BASE_URL": config["BASE_URL"],
                "API_KEY": config["API_KEY"], 
                "MODEL": config["MODEL"],
                "LLM_GATEWAY": config["LLM_GATEWAY"]
            }
        except ValueError as e:
            pytest.skip(f"Environment configuration not available: {e}")
    
    @pytest.fixture
    def test_messages(self):
        """Standard test messages for model testing."""
        return [{"role": "user", "content": vanilla_prompt}]
    
    @pytest.fixture
    def smolagents_model(self, env_config):
        """Initialize smolagents LiteLLMModel."""
        return LiteLLMModel(
            model_id=f"{env_config['LLM_GATEWAY']}/{env_config['MODEL']}",
            api_base=env_config["BASE_URL"],
            api_key=env_config["API_KEY"],
            temperature=0.7
        )
    
    @pytest.fixture
    def backoff_model(self, env_config):
        """Initialize LiteLLMModelWithBackOff."""
        return LiteLLMModelWithBackOff(
            model_id=f"{env_config['LLM_GATEWAY']}/{env_config['MODEL']}",
            api_base=env_config["BASE_URL"],
            api_key=env_config["API_KEY"],
            temperature=0.7,
            max_tokens=1500
        )
    
    def test_smolagents_litellm_model_initialization(self, smolagents_model, env_config):
        """Test that smolagents LiteLLMModel initializes correctly."""
        assert smolagents_model is not None
        assert smolagents_model.model_id == f"litellm_proxy/{env_config['MODEL']}"
        assert smolagents_model.api_base == env_config["BASE_URL"]
    
    def test_backoff_model_initialization(self, backoff_model, env_config):
        """Test that LiteLLMModelWithBackOff initializes correctly."""
        assert backoff_model is not None
        assert backoff_model.model_id == f"litellm_proxy/{env_config['MODEL']}"
        assert backoff_model.api_base == env_config["BASE_URL"]
        assert backoff_model.max_tokens == 1500
    
    def test_smolagents_model_response(self, smolagents_model, test_messages, env_config):
        """Test smolagents LiteLLMModel can generate responses."""
        try:
            response = smolagents_model(test_messages)
            
            # Basic response validation
            assert response is not None
            assert len(str(response)) > 0
            
            print(f"\n--- Smolagents Model Response ---")
            print(f"Model: {env_config['MODEL']}")
            print(f"Response: {response}")
            
        except Exception as e:
            pytest.fail(f"Smolagents model failed to generate response: {e}")
    
    def test_backoff_model_response(self, backoff_model, test_messages, env_config):
        """Test LiteLLMModelWithBackOff can generate responses."""
        try:
            response = backoff_model(test_messages)
            
            # Basic response validation
            assert response is not None
            assert len(str(response)) > 0
            
            print(f"\n--- Backoff Model Response ---")
            print(f"Model: {env_config['MODEL']}")
            print(f"Max tokens: {backoff_model.max_tokens}")
            print(f"Response: {response}")
            
        except Exception as e:
            pytest.fail(f"Backoff model failed to generate response: {e}")
    
 