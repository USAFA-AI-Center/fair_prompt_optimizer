# TESTING_GUIDE.md

# Testing Guide for fair_prompt_optimizer

This guide covers how to test the fair_prompt_optimizer package at different levels.

## Quick Start

```bash
# Install in development mode
cd fair_prompt_optimizer
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v
pytest tests/test_metrics.py -v
pytest tests/test_integration.py -v
```

---

## 1. Unit Tests (No LLM Required)

These tests verify the config and metrics modules work correctly.

```bash
# Run all unit tests
pytest tests/test_config.py tests/test_metrics.py -v

# Expected output: All tests should pass
```

---

## 2. Integration Tests (Requires fairlib)

These tests verify integration with fairlib's PromptBuilder.

```bash
# Make sure fairlib is installed
pip install -e /path/to/fair-llm

# Run integration tests
pytest tests/test_integration.py -v
```

---

## 3. Manual Testing with CLI

### Step 3.1: Test `init` command

```bash
# Create an agent config
fair-optimize init --type agent --output test_agent.json

# Create a simple_llm config
fair-optimize init --type simple_llm --output test_llm.json

# Create a multi_agent config
fair-optimize init --type multi_agent --output test_multi.json

# Verify files were created
ls -la test_*.json
```

### Step 3.2: Test `examples` command

```bash
# Create training examples template
fair-optimize examples --output test_examples.json --count 5

# Verify
cat test_examples.json
```

### Step 3.3: Test `info` command

```bash
# View config info
fair-optimize info -c test_agent.json

# Verbose mode
fair-optimize info -c test_agent.json --verbose
```

---

## 4. Manual Testing with Python API

### Step 4.1: Test Config Module

```python
# test_config_manual.py
from fair_prompt_optimizer.config import (
    OptimizedConfig,
    TrainingExample,
    OptimizationProvenance,
    save_optimized_config,
    load_optimized_config,
)

# Create a config
config = OptimizedConfig(config={
    "version": "1.0",
    "type": "agent",
    "prompts": {
        "role_definition": "You are a test assistant.",
        "tool_instructions": [],
        "format_instructions": ["Be helpful"],
        "examples": [],
    },
    "model": {"adapter": "HuggingFaceAdapter", "model_name": "test"},
    "agent": {"planner_type": "SimpleReActPlanner", "tools": [], "max_steps": 5},
})

# Test accessors
print(f"Role: {config.role_definition}")
print(f"Examples: {config.examples}")
print(f"Type: {config.type}")

# Test provenance
config.optimization.record_run(
    optimizer="bootstrap",
    metric="test_metric",
    num_examples=10,
    examples_before=0,
    examples_after=3,
)
print(f"Optimized: {config.optimization.optimized}")
print(f"Runs: {len(config.optimization.runs)}")

# Save and load
save_optimized_config(config, "test_manual.json")
loaded = load_optimized_config("test_manual.json")
print(f"Loaded role: {loaded.role_definition}")
print(f"Loaded optimized: {loaded.optimization.optimized}")

print("\n✓ Config module works!")
```

Run:
```bash
python test_config_manual.py
```

### Step 4.2: Test Metrics Module

```python
# test_metrics_manual.py
from dataclasses import dataclass
from fair_prompt_optimizer.metrics import (
    exact_match,
    contains_answer,
    numeric_accuracy,
    format_compliance,
    keyword_match,
    create_metric,
)

@dataclass
class Example:
    expected_output: str

@dataclass
class Prediction:
    response: str

# Test exact_match
ex = Example("42")
pred = Prediction("42")
print(f"exact_match('42', '42'): {exact_match(ex, pred)}")  # True

pred = Prediction("The answer is 42")
print(f"exact_match('42', 'The answer is 42'): {exact_match(ex, pred)}")  # False

# Test contains_answer
print(f"contains_answer('42', 'The answer is 42'): {contains_answer(ex, pred)}")  # True

# Test numeric_accuracy
ex = Example("3.14")
pred = Prediction("Pi is approximately 3.14159")
print(f"numeric_accuracy('3.14', '...3.14159'): {numeric_accuracy(ex, pred, tolerance=0.01)}")  # True

# Test format_compliance
metric = format_compliance("ANSWER:")
ex = Example("")
pred = Prediction("ANSWER: 42")
print(f"format_compliance('ANSWER:'): {metric(ex, pred)}")  # True

# Test create_metric
metric = create_metric(check_format="RESULT:", check_numeric=True)
ex = Example("42")
pred = Prediction("RESULT: 42")
print(f"custom metric: {metric(ex, pred)}")  # True

print("\n✓ Metrics module works!")
```

Run:
```bash
python test_metrics_manual.py
```

---

## 5. End-to-End Testing (Requires LLM)

This tests the full optimization flow with a real LLM.

### Step 5.1: Create Test Config

Create `e2e_agent.json`:
```json
{
  "version": "1.0",
  "type": "agent",
  "prompts": {
    "role_definition": "You are a helpful math assistant. Use the calculator tool to solve math problems.",
    "tool_instructions": [
      {"name": "safe_calculator", "description": "Evaluates mathematical expressions safely."}
    ],
    "format_instructions": [
      "Always show your reasoning in a 'Thought' section.",
      "Provide actions as JSON with 'tool_name' and 'tool_input'."
    ],
    "examples": []
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b"
  },
  "agent": {
    "planner_type": "SimpleReActPlanner",
    "tools": ["SafeCalculatorTool"],
    "max_steps": 10
  }
}
```

### Step 5.2: Create Training Examples

Create `e2e_examples.json`:
```json
[
  {"inputs": {"user_input": "What is 25% of 80?"}, "expected_output": "20"},
  {"inputs": {"user_input": "Calculate 144 / 12"}, "expected_output": "12"},
  {"inputs": {"user_input": "What is 15 + 27?"}, "expected_output": "42"},
  {"inputs": {"user_input": "Compute 8 * 7"}, "expected_output": "56"},
  {"inputs": {"user_input": "What is the square root of 16?"}, "expected_output": "4"}
]
```

### Step 5.3: Run Optimization via CLI

```bash
fair-optimize optimize \
  -c e2e_agent.json \
  -t e2e_examples.json \
  --optimizer bootstrap \
  --metric numeric_accuracy \
  --max-demos 2 \
  --output e2e_agent_optimized.json
```

### Step 5.4: Verify Results

```bash
# Check the optimized config
fair-optimize info -c e2e_agent_optimized.json

# Compare before and after
fair-optimize compare e2e_agent.json e2e_agent_optimized.json

# Test interactively
fair-optimize test -c e2e_agent_optimized.json
```

### Step 5.5: Run via Python API

```python
# test_e2e.py
import asyncio
from fairlib import HuggingFaceAdapter
from fairlib.utils.config_manager import load_agent
from fair_prompt_optimizer import (
    AgentOptimizer,
    load_training_examples,
    load_optimized_config,
    numeric_accuracy,
)

async def main():
    # Create LLM
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # Load agent
    agent = load_agent("e2e_agent.json", llm)
    print(f"Loaded agent with {len(agent.planner.prompt_builder.examples)} examples")
    
    # Load training examples
    examples = load_training_examples("e2e_examples.json")
    print(f"Loaded {len(examples)} training examples")
    
    # Create optimizer
    optimizer = AgentOptimizer(agent)
    
    # Run optimization
    print("Running optimization...")
    result = optimizer.compile(
        training_examples=examples,
        metric=numeric_accuracy,
        optimizer="bootstrap",
        max_bootstrapped_demos=2,
        training_data_path="e2e_examples.json",
    )
    
    # Save
    result.save("e2e_agent_optimized_py.json")
    print(f"Saved with {len(result.examples)} examples")
    print(f"Optimization runs: {len(result.optimization.runs)}")
    
    # Test
    print("\nTesting optimized agent...")
    response = optimizer.test("What is 50% of 200?")
    print(f"Response: {response}")
    
    # Load back into fairlib
    optimized_agent = load_agent("e2e_agent_optimized_py.json", llm)
    print(f"\nLoaded optimized agent with {len(optimized_agent.planner.prompt_builder.examples)} examples")

if __name__ == "__main__":
    asyncio.run(main())
```

Run:
```bash
python test_e2e.py
```

---

## 6. Testing Checklist

### Config Module
- [ ] `OptimizedConfig` creation and accessors
- [ ] `OptimizationProvenance.record_run()`
- [ ] `save_optimized_config()` / `load_optimized_config()`
- [ ] `load_training_examples()`
- [ ] `compute_file_hash()`
- [ ] Round-trip serialization

### Metrics Module
- [ ] `exact_match`
- [ ] `contains_answer`
- [ ] `numeric_accuracy` with tolerance
- [ ] `format_compliance` factory
- [ ] `keyword_match` factory
- [ ] `combined_metric` factory
- [ ] `create_metric` factory

### Optimizers
- [ ] `SimpleLLMOptimizer.compile()`
- [ ] `AgentOptimizer.compile()`
- [ ] `MultiAgentOptimizer.compile()`
- [ ] `.test()` method on each optimizer
- [ ] `.save()` method on each optimizer

### CLI
- [ ] `fair-optimize init`
- [ ] `fair-optimize examples`
- [ ] `fair-optimize info`
- [ ] `fair-optimize compare`
- [ ] `fair-optimize optimize`
- [ ] `fair-optimize test`

### Integration
- [ ] Config saved by optimizer loads in fairlib
- [ ] Config saved by fairlib loads in optimizer
- [ ] Optimized PromptBuilder round-trips correctly
- [ ] Provenance preserved across iterations

---

## 7. Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'fairlib'
```
**Fix:** Install fairlib: `pip install -e /path/to/fair-llm`

```
ModuleNotFoundError: No module named 'dspy'
```
**Fix:** Install dspy: `pip install dspy-ai`

### CLI Not Found

```
command not found: fair-optimize
```
**Fix:** Install in dev mode: `pip install -e .`

### Config Loading Errors

```
KeyError: 'prompts'
```
**Fix:** Your config might be in old format. Ensure it has a `prompts` section.

### Optimization Hangs

The optimization might take a while with real LLMs. Try:
- Reduce `--max-demos` to 1 or 2
- Use a smaller/faster model
- Check your LLM is responding

---

## 8. Running All Tests

```bash
# Full test suite
cd fair_prompt_optimizer
pip install -e ".[dev]"

# Unit tests (fast, no LLM)
pytest tests/test_config.py tests/test_metrics.py -v

# Integration tests (requires fairlib)
pytest tests/test_integration.py -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=fair_prompt_optimizer --cov-report=html
```