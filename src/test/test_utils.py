import os

TEST_PROMPT = "This is a connectivity test. If you receive this message, reply I'm here! with a hand emoji. After that tell a dad joke about computers"

def get_env():
    # --- Configuration & Validation ---
    BASE_URL = os.getenv("BASE_URL")
    API_KEY = os.getenv("API_KEY")
    MODEL = os.getenv("MODEL")
    SSL_CERT_FILE = os.getenv("SSL_CERT_FILE")

    if not all([BASE_URL, API_KEY, MODEL, SSL_CERT_FILE]):
        raise ValueError("One or more required environment variables (BASE_URL, API_KEY, MODEL, SSL_CERT_FILE) are not set in your .env file.")

    return BASE_URL, API_KEY, MODEL