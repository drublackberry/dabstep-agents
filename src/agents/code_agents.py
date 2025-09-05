from smolagents import CodeAgent, OpenAIServerModel
from agents.models import LiteLLMModelWithBackOff
from agents.prompts import reasoning_llm_system_prompt
from constants import ADDITIONAL_AUTHORIZED_IMPORTS

# Filter out markdown from authorized imports to avoid missing dependency error
FILTERED_AUTHORIZED_IMPORTS = [imp for imp in ADDITIONAL_AUTHORIZED_IMPORTS if imp != 'markdown']


def read_only_open(*a, **kw):
    """Restricted open function that only allows read mode."""
    if (len(a) > 1 and isinstance(a[1], str) and a[1] != 'r') or kw.get('mode', 'r') != 'r':
        raise Exception("Only mode='r' allowed for the function open")
    return open(*a, **kw)


class CustomCodeAgent(CodeAgent):
    
    def __init__(self, *args, **kwargs):
        # Set a default system prompt template before calling super().__init__
        self.system_prompt_template = "You are a helpful assistant."
        super().__init__(*args, **kwargs)
        # Update with actual system prompt after initialization
        if hasattr(self, 'system_prompt') and self.system_prompt:
            self.system_prompt_template = self.system_prompt

    def initialize_system_prompt(self):
        return self.system_prompt_template


class ReasoningCodeAgent(CustomCodeAgent):
    """A specialized CodeAgent configured for reasoning LLMs."""
    
    def __init__(self, model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None):
        super().__init__(
            tools=[],
            model=LiteLLMModelWithBackOff(
                model_id=model_id, api_base=api_base, api_key=api_key, max_tokens=None, max_completion_tokens=3000),
            additional_authorized_imports=FILTERED_AUTHORIZED_IMPORTS,
            max_steps=max_steps,
            verbosity_level=3,
        )
        
        # Set system prompt after initialization using prompt_templates
        self.prompt_templates["system_prompt"] = reasoning_llm_system_prompt
        
        # Set up read-only file access
        if hasattr(self, 'python_executor') and self.python_executor is not None:
            self.python_executor.static_tools.update({"open": read_only_open})
        
        # Format system prompt with context path
        if ctx_path:
            self.prompt_templates["system_prompt"] = self.prompt_templates["system_prompt"].format(ctx_path=ctx_path)


class ChatCodeAgent(CodeAgent):
    """A specialized CodeAgent configured for chat LLMs."""
    
    def __init__(self, model_id: str, api_base=None, api_key=None, max_steps=10):
        super().__init__(
            tools=[],
            model=OpenAIServerModel(model_id=model_id, api_base=api_base, api_key=api_key, max_tokens=3000),
            additional_authorized_imports=FILTERED_AUTHORIZED_IMPORTS,
            max_steps=max_steps,
            verbosity_level=3,
            executor_type="local",
        )

