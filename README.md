# fair_prompt_optimizer

DSPy-powered prompt optimization for FAIR-LLM agents.

## Overview

This tool optimizes FAIR-LLM agent prompts by running your **actual agent** (with tools, memory, etc.) on training examples and using DSPy's optimization algorithms to improve performance.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION FLOW                            │
│                                                                 │
│  Training Examples                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │     Your FAIR-LLM Component             │                    │
│  │  • SimpleLLM (classification, etc.)     │                    │
│  │  • Agent (tools + ReAct)                │                    │
│  │  • Multi-Agent (manager + workers)      │                    │
│  └─────────────────────────────────────────┘                    │
│            │                                                    │
│            ▼                                                    │
│     Response ──► Metric ──► DSPy Optimizer                      │
│                                        │                        │
│                                        ▼                        │
│                              Optimized Config                   │
│                              (examples + instructions)          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone git@github.com:USAFA-AI-Center/fair_prompt_optimizer.git
cd fair_prompt_optimizer
pip install -r requirements.txt
```

**Requires** `fairlib` to be installed:
```bash
pip install fair-llm
```

## Three Levels of Optimization

| Level | Optimizer | Use Case | Training Data |
|-------|-----------|----------|---------------|
| 1 | `SimpleLLMOptimizer` | Classification, format compliance | `inputs` + `expected_output` |
| 2 | `AgentOptimizer` | Tool-using agents, ReAct | + `full_trace` (required) |
| 3 | `MultiAgentOptimizer` | Manager + workers | + `full_trace` (required) |

**A full trace example should be identical to a loop through the fair_llm framework, look at the training data in the examples directory**
## Quick Start (CLI)

```bash
# 1. Create starter config files
fair-optimize init --type agent

# 2. Edit agent_config.json and create training examples

# 3. Run optimization with BootstrapFewShot
fair-optimize optimize -c agent_config.json -t examples.json --optimizer bootstrap

# 4. Test your optimized agent
fair-optimize test -c agent_config_optimized.json
```

### Using MIPROv2 (Optimizes Instructions + Examples)

MIPROv2 requires an LLM for instruction generation:

```bash
# With Ollama (local)
fair-optimize optimize \
    -c agent_config.json \
    -t examples.json \
    --optimizer mipro \
    --mipro-lm "ollama_chat/llama3:8b" \
    --mipro-auto medium

# With OpenAI
export OPENAI_API_KEY="sk-..."
fair-optimize optimize \
    -c agent_config.json \
    -t examples.json \
    --optimizer mipro \
    --mipro-lm "openai/gpt-4o-mini"
```

## Quick Start (Python API)

### Level 1: SimpleLLMOptimizer

For classification, format compliance, simple generation—no agent pipeline.

```python
from fairlib import HuggingFaceAdapter
from fair_prompt_optimizer import (
    SimpleLLMOptimizer,
    load_training_examples,
    format_compliance,
)

# 1. Create LLM and system prompt
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
system_prompt = """You are a sentiment classifier.
Respond with ONLY: SENTIMENT: positive, negative, or neutral"""

# 2. Load training examples (no full_trace needed)
examples = load_training_examples("sentiment_examples.json")

# 3. Optimize
optimizer = SimpleLLMOptimizer(llm, system_prompt)
config = optimizer.compile(
    training_examples=examples,
    metric=format_compliance("SENTIMENT:"),
    optimizer="bootstrap",
    max_bootstrapped_demos=3,
)

# 4. Save and test
config.save("classifier_optimized.json")
response = optimizer.test("I love this product!")
```

### Level 2: AgentOptimizer

For tool-using agents with ReAct planning.

```python
from fairlib import (
    HuggingFaceAdapter,
    SimpleAgent,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
)
from fairlib.modules.planning.react_planner import ReActPlanner
from fairlib.core.prompts import PromptBuilder, RoleDefinition
from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
from fairlib.utils.config_manager import save_agent_config, load_agent

from fair_prompt_optimizer import (
    AgentOptimizer,
    load_training_examples,
    numeric_accuracy,
)

# 1. Build your agent
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
tool_registry = ToolRegistry()
tool_registry.register_tool(SafeCalculatorTool())

prompt_builder = PromptBuilder()
prompt_builder.role_definition = RoleDefinition(
    "You are a math assistant. Use safe_calculator for ALL calculations."
)

planner = ReActPlanner(llm, tool_registry, prompt_builder=prompt_builder)
agent = SimpleAgent(
    llm=llm,
    planner=planner,
    tool_executor=ToolExecutor(tool_registry),
    memory=WorkingMemory(),
    max_steps=10,
)

# 2. Load training examples (with full_trace for best results)
examples = load_training_examples("math_examples.json")

# 3. Optimize
optimizer = AgentOptimizer(agent)
config = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="bootstrap",
    max_bootstrapped_demos=4,
)

# 4. Save
config.save("agent_optimized.json")

# 5. Load and use optimized agent
optimized_agent = load_agent("agent_optimized.json", llm)
result = await optimized_agent.arun("What is 75% of 120?")
```

### Level 3: MultiAgentOptimizer

For hierarchical systems with manager and workers.

```python
from fairlib.modules.orchestration.hierarchical_runner import HierarchicalAgentRunner
from fairlib.utils.config_manager import load_multi_agent

from fair_prompt_optimizer import (
    MultiAgentOptimizer,
    load_training_examples,
    contains_answer,
)

# 1. Create runner (manager + workers)
runner = HierarchicalAgentRunner(
    manager=manager_agent,
    workers={"Calculator": calculator_worker},
)

# 2. Load examples (full_trace required)
examples = load_training_examples("multi_agent_examples.json")

# 3. Optimize
optimizer = MultiAgentOptimizer(runner)
config = optimizer.compile(
    training_examples=examples,
    metric=contains_answer,
    optimizer="bootstrap",
    optimize_manager=True,
    optimize_workers=False,
    max_bootstrapped_demos=2,
)

# 4. Save and load
config.save("multi_agent_optimized.json")
optimized_runner = load_multi_agent("multi_agent_optimized.json", llm)
```

## Optimizers

### BootstrapFewShot

Generates few-shot demos from successful execution traces.

**What it optimizes:** Examples only (not instructions)

**Best for:**
- Quick iterations
- Small datasets (10-50 examples)
- When your instructions are already good

```python
config = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="bootstrap",
    max_bootstrapped_demos=4,
)
```

### MIPROv2

Uses Bayesian optimization to find the best combination of instructions AND examples.

**What it optimizes:** Instructions + Examples

**Best for:**
- Production optimization
- Larger datasets (50+ examples)
- When you want better instructions

**Requires:** A DSPy-compatible LM for instruction generation

```python
import dspy

dspy_lm = dspy.LM('ollama_chat/llama3:8b', api_base='http://localhost:11434')

config = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="mipro",
    dspy_lm=dspy_lm,
    mipro_auto="medium",  # light, medium, or heavy
    max_bootstrapped_demos=4,
)
```

| Mode | Trials | Speed | Use Case |
|------|--------|-------|----------|
| `light` | ~10 | Fast | Testing, quick iteration |
| `medium` | ~25 | Moderate | Balanced optimization |
| `heavy` | ~50+ | Slow | Final production tuning |

## Training Data Format

### For SimpleLLM (no full_trace needed)

```json
[
  {
    "inputs": {"user_input": "I love this product!"},
    "expected_output": "SENTIMENT: positive"
  },
  {
    "inputs": {"user_input": "Terrible experience."},
    "expected_output": "SENTIMENT: negative"
  }
]
```

### For Agents (with full_trace)

```json
[
  {
    "inputs": {"user_input": "What is 25 percent of 80?"},
    "expected_output": "20",
    "full_trace": "# --- Example ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"0.25 * 80\"\n    }\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"20\"\n    }\n}"
  }
]
```

The `full_trace` shows the complete ReAct loop so the model learns the workflow pattern.

## Built-in Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| `exact_match` | Exact string match | `metric=exact_match` |
| `contains_answer` | Expected is substring of actual | `metric=contains_answer` |
| `numeric_accuracy` | Numeric comparison (1% tolerance) | `metric=numeric_accuracy` |
| `fuzzy_match` | Character-level similarity | `metric=fuzzy_match` |
| `format_compliance(prefix)` | Output starts with prefix | `metric=format_compliance("ANSWER:")` |

### Custom Metrics

```python
def my_metric(example, prediction, trace=None) -> bool:
    expected = example.expected_output.lower()
    actual = prediction.response.lower()
    return expected in actual

config = optimizer.compile(..., metric=my_metric)
```

## Config Files

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for complete documentation of all config structures.

### Quick Overview

```json
{
  "version": "1.0",
  "type": "simple_llm | agent | multi_agent",
  "prompts": {
    "role_definition": "...",
    "examples": ["..."]
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b"
  },
  "optimization": {
    "runs": [...]
  }
}
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `fair-optimize init` | Create starter config files |
| `fair-optimize optimize` | Run prompt optimization |
| `fair-optimize test` | Test an agent interactively |
| `fair-optimize info` | Show config information |
| `fair-optimize compare` | Compare two configs |
| `fair-optimize examples` | Create example training data template |

### Optimize Options

```
fair-optimize optimize [OPTIONS]

Required:
  -c, --config PATH         Agent config JSON
  -t, --training PATH       Training examples JSON

Optional:
  -o, --output PATH         Output path (default: {config}_optimized.json)
  --optimizer [bootstrap|mipro]   Algorithm (default: bootstrap)
  --metric [exact|contains|numeric|fuzzy]   Metric (default: contains)
  --max-demos INT           Max demos to generate (default: 4)
  --mipro-lm TEXT           DSPy LM for MIPROv2 (e.g., 'ollama_chat/llama3:8b')
  --mipro-auto [light|medium|heavy]   MIPROv2 intensity (default: light)
```

## Iterative Optimization

You can run multiple optimization passes, building on previous results:

```python
from fair_prompt_optimizer import load_optimized_config, AgentOptimizer

# Load previous optimization
config = load_optimized_config("agent_optimized.json")
agent = load_agent("agent_optimized.json", llm)

# Run another round with new examples
optimizer = AgentOptimizer(agent, config=config)
config = optimizer.compile(
    training_examples=new_examples,
    metric=numeric_accuracy,
    optimizer="mipro",
    dspy_lm=dspy_lm,
)

config.save("agent_optimized_v2.json")

# Check optimization history
print(f"Total runs: {len(config.optimization.runs)}")
```

## Troubleshooting

### MIPROv2 requires a DSPy LM

```
ValueError: MIPROv2 requires dspy_lm parameter
```

**Solution:** Provide a DSPy LM:
```python
import dspy
dspy_lm = dspy.LM('ollama_chat/llama3:8b', api_base='http://localhost:11434')
optimizer.compile(..., dspy_lm=dspy_lm)
```

### Ollama connection refused

```
ConnectionRefusedError: Connection refused
```

**Solution:** Start Ollama server:
```bash
ollama serve
ollama pull llama3:8b
```

### No examples selected during bootstrap

This happens when the agent fails the metric on all training examples.

**Solutions:**
- Check that your `expected_output` matches what the agent actually produces
- Use a more lenient metric (e.g., `contains_answer` instead of `exact_match`)
- Ensure `full_trace` format matches your agent's output format

### Dict is not of type OptimizedConfig

```
TypeError: expected OptimizedConfig, got dict
```

**Solution:** Use `load_optimized_config` instead of `load_agent_config`:
```python
from fair_prompt_optimizer import load_optimized_config

config = load_optimized_config("agent_optimized.json")  # ✓
# NOT: load_agent_config("agent_optimized.json")        # ✗
```

## Examples

See `examples/examples.py` for complete working examples:

```bash
cd examples
python examples.py
```

## API Reference

### Config Functions

```python
from fair_prompt_optimizer import (
    # Loading/saving
    load_optimized_config,
    save_optimized_config,
    load_training_examples,
    save_training_examples,
    
    # Config classes
    OptimizedConfig,
    TrainingExample,
    OptimizationProvenance,
)
```

### Optimizer Classes

```python
from fair_prompt_optimizer import (
    SimpleLLMOptimizer,
    AgentOptimizer,
    MultiAgentOptimizer,
)
```

### Metrics

```python
from fair_prompt_optimizer import (
    exact_match,
    contains_answer,
    numeric_accuracy,
    fuzzy_match,
    format_compliance,
    keyword_match,
    combined_metric,
    create_metric,
)
```