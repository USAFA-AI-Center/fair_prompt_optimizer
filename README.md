# fair_prompt_optimizer

DSPy-powered prompt optimization for FAIR-LLM agents.

## Overview

This tool optimizes FAIR-LLM agent prompts by running your **actual agent** (with tools, memory, etc.) on training examples and using DSPy's optimization algorithms to improve performance.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION FLOW                            │
│                                                                 │
│  Training Example                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │        Your FAIR-LLM Agent              │                    │
│  │  • LLM                                  │                    │
│  │  • Tools (calculator, RAG, etc.)        │                    │
│  │  • Memory                               │                    │
│  │  • ReAct Loop                           │                    │
│  └─────────────────────────────────────────┘                    │
│            │                                                    │
│            ▼                                                    │
│     Agent Response ──► Metric ──► DSPy Optimizer                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone git@github.com:USAFA-AI-Center/fair_prompt_optimizer.git
cd fair_prompt_optimizer
pip install -r requirements.txt
```

**Note:** Requires `fair_llm` to be installed:
```bash
pip install fair-llm
```

## Quick Start (CLI)

The easiest way to use the optimizer is via the command line:

```bash
# 1. Create starter config files
fair-optimize init

# 2. Edit agent_config.json and examples.json to your needs

# 3. Run optimization with BootstrapFewShot (no external LM needed)
fair-optimize -c agent_config.json -t examples.json --optimizer bootstrap

# 4. Test your optimized agent
fair-optimize test -c agent_config_optimized.json
```

### Using MIPROv2 with Ollama (Local Models)

MIPROv2 optimizes instructions and requires an LLM for instruction generation:

```bash
# Terminal 1: Start Ollama
ollama serve
ollama pull llama3:8b

# Terminal 2: Run MIPROv2 optimization
fair-optimize \
    -c agent_config.json \
    -t examples.json \
    --optimizer mipro \
    --ollama-model llama3:8b \
    --mipro-mode light
```

### Using MIPROv2 with OpenAI

```bash
export OPENAI_API_KEY="sk-..."

fair-optimize \
    -c agent_config.json \
    -t examples.json \
    --optimizer mipro \
    --openai-model gpt-4o-mini
```

## Quick Start (Python API)

```python
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition
)
from fair_prompt_optimizer import (
    FAIRPromptOptimizer,
    TrainingExample,
    numeric_accuracy,
    load_optimized_agent,
)

# 1. Build your agent as usual
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
tool_registry = ToolRegistry()
tool_registry.register_tool(SafeCalculatorTool())
executor = ToolExecutor(tool_registry)
memory = WorkingMemory()
planner = SimpleReActPlanner(llm, tool_registry)
planner.prompt_builder.role_definition = RoleDefinition("You are a calculator...")

agent = SimpleAgent(llm, planner, executor, memory, max_steps=10)

# 2. Create training examples
examples = [
    TrainingExample(inputs={"user_input": "What is 15 + 27?"}, expected_output="42"),
    TrainingExample(inputs={"user_input": "Calculate 100 / 4"}, expected_output="25"),
]

# 3. Optimize with BootstrapFewShot
optimizer = FAIRPromptOptimizer(agent)
config = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="bootstrap",
    output_path="optimized_config.json"
)

# 4. Later: Load optimized agent from config
agent = load_optimized_agent("optimized_config.json")
response = await agent.arun("What is 15 + 27?")
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `fair-optimize init` | Create starter config files |
| `fair-optimize optimize` | Run prompt optimization |
| `fair-optimize test` | Test an agent interactively |
| `fair-optimize info` | Show help and examples |

### Optimize Options

```
fair-optimize optimize [OPTIONS]

Options:
  -c, --config PATH              Agent config JSON (required)
  -t, --training-data PATH       Training examples JSON (required)
  -o, --output PATH              Output path (default: <input>_optimized.json)
  --optimizer [bootstrap|mipro]  Algorithm (default: bootstrap)
  --metric [exact|contains|numeric|fuzzy]  Metric (default: numeric)
  --max-demos INT                Max demos to generate (default: 4)
  --mipro-mode [light|medium|heavy]  MIPROv2 intensity (default: light)
  --ollama-model TEXT            Ollama model for MIPROv2
  --ollama-url TEXT              Ollama URL (default: http://localhost:11434)
  --openai-model TEXT            OpenAI model for MIPROv2
  -v, --verbose                  Enable verbose output
```

## Optimizers

### BootstrapFewShot

Generates demos from successful agent execution traces.

**Best for:**
- Quick iterations
- Small datasets (10-50 examples)
- Demo-based learning
- No external LM needed

```bash
fair-optimize -c config.json -t examples.json --optimizer bootstrap
```

### MIPROv2

Optimizes instruction text using Bayesian optimization. Requires a DSPy-compatible LM.

**Best for:**
- Instruction tuning
- Larger datasets (50+ examples)
- Production optimization

**Modes:**
- `light` - Fast, fewer trials
- `medium` - Balanced
- `heavy` - Thorough, more trials

```bash
fair-optimize -c config.json -t examples.json \
    --optimizer mipro \
    --mipro-mode medium \
    --ollama-model llama3:8b
```

## Config File Format

### Agent Config (`agent_config.json`)

```json
{
  "version": "1.0",
  "role_definition": "You are an expert mathematical calculator...",
  "examples": [],
  "model": {
    "model_name": "dolphin3-qwen25-3b",
    "adapter": "HuggingFaceAdapter",
    "adapter_kwargs": {}
  },
  "agent": {
    "agent_type": "SimpleAgent",
    "planner_type": "SimpleReActPlanner",
    "max_steps": 10,
    "tools": ["SafeCalculatorTool"]
  },
  "metadata": {}
}
```

### Training Examples (`examples.json`)

```json
[
  {
    "inputs": {"user_input": "What is 15 + 27?"},
    "expected_output": "42"
  },
  {
    "inputs": {"user_input": "Calculate 100 divided by 4"},
    "expected_output": "25"
  }
]
```

### Optimized Config (Output)

After optimization, the config includes demos and metadata:

```json
{
  "version": "1.0",
  "role_definition": "You are an expert mathematical calculator...",
  "examples": [
    "User: What is 15 + 27?\nResponse: 42",
    "User: Calculate 100 divided by 4\nResponse: 25"
  ],
  "model": { ... },
  "agent": { ... },
  "metadata": {
    "optimized": true,
    "optimized_at": "2025-12-11T10:30:00",
    "optimizer": "bootstrap",
    "num_training_examples": 8
  }
}
```

## Metrics

Built-in evaluation metrics:

| Metric | Description |
|--------|-------------|
| `exact` | Exact string match (case-insensitive) |
| `contains` | Expected answer is substring of response |
| `numeric` | Numeric comparison with 1% tolerance |
| `fuzzy` | Fuzzy string matching |

### Custom Metrics (Python API)

```python
def my_metric(example, prediction, trace=None) -> bool:
    expected = example.response.lower()
    predicted = prediction.response.lower()
    return expected in predicted

config = optimizer.compile(..., metric=my_metric)
```

## Loading Optimized Agents

After optimization, load your agent from the config file:

```python
from fair_prompt_optimizer import load_optimized_agent

# Load agent with optimized prompts - no manual setup needed!
agent = load_optimized_agent("optimized_config.json")

# Use the agent
response = await agent.arun("What is 15 + 27?")
```

## Examples

See `examples/optimize_fair_agent_bootstrap.py` for a complete working example.

```bash
cd examples
python optimize_fair_agent_bootstrap.py
```

## Troubleshooting

### MIPROv2 requires a DSPy LM

```
ValueError: MIPROv2 requires a DSPy LM for instruction generation.
```

**Solution:** Provide `--ollama-model` or `--openai-model`:
```bash
fair-optimize ... --optimizer mipro --ollama-model llama3:8b
```

### Ollama connection refused

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:** Start Ollama server:
```bash
ollama serve
```

### Model not found in registry

```
ValueError: Unknown tool: MyCustomTool
```

**Solution:** Register custom tools in `registry.py` or use built-in tools.