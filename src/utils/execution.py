"""
General execution utilities for environment configuration and logging.
"""

import os
import logging
from tqdm import tqdm
from dotenv import load_dotenv


def get_env():
    """
    Load and validate environment variables from .env file.
    
    Returns:
        dict: Configuration dictionary with validated environment variables
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # --- Configuration & Validation ---
    config = {
        "BASE_URL": os.getenv("BASE_URL"),
        "API_KEY": os.getenv("API_KEY"),
        "MODEL": os.getenv("MODEL"),
        "LLM_GATEWAY": os.getenv("LLM_GATEWAY"),
        "SSL_CERT_FILE": os.getenv("SSL_CERT_FILE"),
        "HF_TOKEN": os.getenv("HF_TOKEN")
    }
    
    # Check each required variable individually
    required_vars = ["BASE_URL", "API_KEY", "MODEL", "SSL_CERT_FILE", "HF_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not config[var]:
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"The following required environment variables are not set in your .env file: {', '.join(missing_vars)}")
    
    return config


def validate_reasoning_model_compatibility(model_id: str, use_reasoning: bool) -> None:
    """
    Validate that the use_reasoning parameter is compatible with the model.
    
    Args:
        model_id: The model identifier
        use_reasoning: Whether reasoning mode is enabled
        
    Raises:
        Warning: If there's a mismatch between model capabilities and use_reasoning setting
    """
    reasoning_llm_list = [
        "openai/o1",
        "openai/o3",
        "openai/o3-mini",
        "deepseek/deepseek-reasoner"
    ]
    
    is_reasoning_model = model_id in reasoning_llm_list
    
    if is_reasoning_model and not use_reasoning:
        logging.warning(f"Model '{model_id}' is a reasoning model but use_reasoning=False. "
                       f"Consider setting use_reasoning=True for optimal performance.")
    elif not is_reasoning_model and use_reasoning:
        logging.warning(f"Model '{model_id}' is not a reasoning model but use_reasoning=True. "
                       f"Consider setting use_reasoning=False for this model.")


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that works with tqdm progress bars.
    Ensures log messages don't interfere with progress bar display.
    """
    def emit(self, record):
        tqdm.write(self.format(record))
