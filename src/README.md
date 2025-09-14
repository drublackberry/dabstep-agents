# Dabstep Agents

A data analysis agent system built with smolagents that supports both reasoning and chat-based LLMs for comprehensive data analysis tasks.

## Project Structure

```
src/
├── agents/           # Agent implementations and prompts
├── test/            # Test scripts for agent capabilities
├── utils/           # Utilities for execution and tracing
├── run.py           # Main execution script
└── constants.py     # Project constants
```

## Setup

### 1. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# API Configuration
BASE_URL=https://api.your-llm-provider.com/v1
API_KEY=sk-your-api-key-here
MODEL=gpt-4o-mini
LLM_GATEWAY=openai

# Hugging Face Token (for dataset access)
HF_TOKEN=hf_your-huggingface-token-here

# SSL Configuration (if needed)
SSL_CERT_FILE=/path/to/your/cert.pem

# Optional: OTLP Endpoint for custom tracing
OTLP_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Tracing (Optional)

For observability and debugging, start Phoenix tracing server:

```bash
python -m phoenix.server.main serve &
```

Access the Phoenix UI at: http://127.0.0.1:6006

## Usage

### Running the Main Script

The main execution script supports both reasoning and chat-based agents:

```bash
# Run with chat agent (default)
python src/run.py --max-tasks 5 --split dev

# Run with reasoning agent
python src/run.py --use-reasoning --max-tasks 5 --split dev

# Run specific tasks
python src/run.py --tasks-ids 1 5 10 --split dev

# Run all tasks from dev split
python src/run.py --max-tasks -1 --split dev

# Custom model configuration
python src/run.py --model-id gpt-4o --max-steps 15 --split default
```

#### Command Line Arguments

- `--model-id`: Model identifier (default from .env MODEL)
- `--use-reasoning`: Enable reasoning mode (uses ReasoningCodeAgent)
- `--max-tasks`: Number of tasks to run (-1 for all)
- `--max-steps`: Maximum steps per task (default: 10)
- `--split`: Dataset split to use (`dev` or `default`)
- `--tasks-ids`: Specific task IDs to run
- `--api-base`: API base URL (default from .env BASE_URL)
- `--api-key`: API key (default from .env API_KEY)
- `--concurrency`: Number of parallel tasks (default: 1)

**Note:** All configuration values will be automatically loaded from your `.env` file if not explicitly provided via command line arguments. Command line arguments take precedence over `.env` values.

### Running Tests

The project includes several test scripts to validate agent capabilities:

#### Data Analysis Tests

**Chat Agent Test:**
```bash
cd src && python test/test_data_analysis_chat.py
```
- Tests ChatCodeAgent's conversational data analysis capabilities
- Creates synthetic sales data and runs comprehensive analysis
- Traces appear in Phoenix under "chat-analysis" project

**Reasoning Agent Test:**
```bash
cd src && python test/test_data_analysis_reasoning.py
```
- Tests ReasoningCodeAgent's analytical reasoning capabilities  
- Uses the same test data but with reasoning-focused prompts
- Traces appear in Phoenix under "reasoning-analysis" project

#### Security Test

**Read-Only Security Test:**
```bash
cd src && python test/test_read_only_security.py
```
- Validates that agents have read-only file system access
- Ensures security constraints are properly enforced

#### Tracing Debug Test

**Tracing Connectivity Test:**
```bash
cd src && python test/test_tracing_debug.py
```
- Debugs tracing setup and connectivity to Phoenix
- Validates that traces are properly exported
- Traces appear in Phoenix under "debug-test" project

### Test Results

Each test provides detailed output including:
- ✅ Successful operations and validations
- ❌ Failed operations with error details  
- 📊 Analysis results and insights
- 🎯 Overall capability assessment

## Agent Types

### ChatCodeAgent
- Optimized for conversational data analysis
- Uses chat-style prompts with step-by-step guidance
- Best for interactive analysis workflows

### ReasoningCodeAgent  
- Designed for reasoning-capable LLMs (o1, o3, etc.)
- Uses reasoning-focused prompts
- Best for complex analytical tasks requiring deep reasoning

## Tracing and Observability

The system includes comprehensive tracing via OpenTelemetry and Phoenix:

- **Automatic instrumentation** of smolagents operations
- **Project separation** for different test types
- **Real-time monitoring** of agent execution
- **Performance analytics** and debugging capabilities

Access traces at: http://127.0.0.1:6006

## Configuration Examples

### For OpenAI Models
```bash
BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-openai-key
MODEL=gpt-4o-mini
LLM_GATEWAY=openai
```

### For Reasoning Models
```bash
BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-openai-key
MODEL=o3-mini
LLM_GATEWAY=openai
```

### For Custom Providers
```bash
BASE_URL=https://your-provider.com/v1
API_KEY=your-custom-key
MODEL=your-model-name
LLM_GATEWAY=your-provider
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory (`cd src`)
2. **API Errors**: Verify your `.env` configuration and API keys
3. **Tracing Issues**: Check Phoenix server is running and accessible
4. **Permission Errors**: Agents have read-only access by design

### Debug Commands

```bash
# Test environment configuration
python -c "from utils.execution import get_env; print(get_env())"

# Test tracing setup
python test/test_tracing_debug.py

# Validate agent creation
python -c "from agents.code_agents import ChatCodeAgent; print('✅ Import successful')"
```