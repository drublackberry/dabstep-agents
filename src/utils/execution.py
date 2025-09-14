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
        "SSL_CERT_FILE": os.getenv("SSL_CERT_FILE")
    }
    
    # Check each required variable individually
    required_vars = ["BASE_URL", "API_KEY", "MODEL", "SSL_CERT_FILE"]
    missing_vars = []
    
    for var in required_vars:
        if not config[var]:
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"The following required environment variables are not set in your .env file: {', '.join(missing_vars)}")
    
    return config


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that works with tqdm progress bars.
    Ensures log messages don't interfere with progress bar display.
    """
    def emit(self, record):
        tqdm.write(self.format(record))
