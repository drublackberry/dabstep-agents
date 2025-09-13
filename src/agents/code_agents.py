from smolagents import CodeAgent, OpenAIServerModel
from agents.models import LiteLLMModelWithBackOff
from agents.prompts import reasoning_llm_system_prompt
from constants import ADDITIONAL_AUTHORIZED_IMPORTS
from utils.tracing import setup_smolagents_tracing

# Filter out markdown from authorized imports to avoid missing dependency error
FILTERED_AUTHORIZED_IMPORTS = [imp for imp in ADDITIONAL_AUTHORIZED_IMPORTS if imp != 'markdown']


def read_only_open(*args, **kwargs):
    """Restricted open function that only allows read mode."""
    # Check positional mode argument
    if len(args) > 1 and isinstance(args[1], str):
        mode = args[1]
        if 'r' not in mode or any(char in mode for char in 'wax+'):
            raise PermissionError("Only read mode ('r', 'rb', 'rt') is allowed")
    
    # Check keyword mode argument
    mode = kwargs.get('mode', 'r')
    if 'r' not in mode or any(char in mode for char in 'wax+'):
        raise PermissionError("Only read mode ('r', 'rb', 'rt') is allowed")
    
    # Ensure mode is explicitly set to read-only
    if len(args) > 1:
        args = list(args)
        args[1] = 'r' if 'b' not in str(args[1]) else 'rb'
    else:
        kwargs['mode'] = 'r'
    
    return open(*args, **kwargs)


class ReasoningCodeAgent(CodeAgent):
    """A specialized CodeAgent configured for reasoning LLMs with read-only access."""
    
    def __init__(self, model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None, enable_tracing=True):
        # Set up tracing before agent initialization
        setup_smolagents_tracing(enable_tracing=enable_tracing)
        
        # Initialize the parent CodeAgent without system_prompt parameter
        super().__init__(
            tools=[],
            model=LiteLLMModelWithBackOff(
                model_id=model_id, api_base=api_base, api_key=api_key, max_tokens=None, max_completion_tokens=3000),
            additional_authorized_imports=FILTERED_AUTHORIZED_IMPORTS,
            max_steps=max_steps,
            verbosity_level=3,
        )
        
        # Format and set system prompt after initialization
        formatted_prompt = reasoning_llm_system_prompt
        if ctx_path:
            formatted_prompt = reasoning_llm_system_prompt.format(ctx_path=ctx_path, authorized_imports=FILTERED_AUTHORIZED_IMPORTS)
        
        # Set the system prompt using the prompt_templates approach
        self.prompt_templates["system_prompt"] = formatted_prompt
        
        # Store the read_only_open function for later use
        self._read_only_open = read_only_open
        
        # Override the python executor initialization to ensure read-only access
        self._setup_read_only_executor()
    
    def _setup_read_only_executor(self):
        """Set up comprehensive read-only file access."""
        if (hasattr(self, 'python_executor') and 
            self.python_executor is not None and 
            hasattr(self.python_executor, 'static_tools') and
            self.python_executor.static_tools is not None):
            
            # Override file operations with read-only versions
            restricted_tools = {
                "open": read_only_open,
                # Block other potentially dangerous file operations
                "exec": lambda *args, **kwargs: self._block_operation("exec"),
                "eval": lambda *args, **kwargs: self._block_operation("eval"), 
                "compile": lambda *args, **kwargs: self._block_operation("compile"),
            }
            self.python_executor.static_tools.update(restricted_tools)
        else:
            # If executor isn't ready yet, we'll set it up later in _setup_executor
            pass
    
    def _block_operation(self, operation_name):
        """Block potentially dangerous operations."""
        raise PermissionError(f"Operation '{operation_name}' is not allowed in read-only mode")
            
    def _setup_executor(self):
        """Override executor setup to ensure read-only access is maintained."""
        super()._setup_executor()
        # Override open function in the executor's global namespace
        if hasattr(self, 'python_executor') and self.python_executor is not None:
            self.python_executor.globals['open'] = self._read_only_open
        self._setup_read_only_executor()


