from typing import Optional

from smolagents import LiteLLMModel
from tenacity import retry, stop_after_attempt, before_sleep_log, retry_if_exception_type, wait_exponential, wait_random
import litellm
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class LiteLLMModelWithBackOff(LiteLLMModel):
    def __init__(self, max_tokens: Optional[int] = 1500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(450),
        wait=wait_exponential(min=1, max=120, exp_base=2, multiplier=1) + wait_random(0, 5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type((
                litellm.Timeout,
                litellm.RateLimitError,
                litellm.APIConnectionError,
                litellm.InternalServerError
        ))
    )
    def __call__(self, *args, **kwargs):
        return super().__call__(max_tokens=self.max_tokens, *args, **kwargs)

