# fair_prompt_optimizer

A prompt optimization utility for FAIR-LLM that leverages DSPy's optimization algorithms to automatically improve agent prompts.

## Overview

This tool bridges FAIR-LLM's `PromptBuilder` with DSPy's powerful prompt optimizers (BootstrapFewShot, MIPROv2) without introducing any DSPy dependencies into FAIR-LLM itself.

```
┌─────────────────────┐         JSON          ┌─────────────────────┐
│      FAIR-LLM       │ ◄──────────────────►  │fair_prompt_optimizer│
│  (no DSPy deps)     │    (the contract)     │   (uses DSPy)       │
└─────────────────────┘                       └─────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fair_prompt_optimizer.git
cd fair_prompt_optimizer

# Install with pip
pip install -r requirements.txt
```

## Quick Start

### 1. Export your FAIR-LLM prompts

```python
# In your FAIR-LLM code
from fairlib import SimpleReActPlanner

planner = SimpleReActPlanner(llm, tool_registry)
planner.prompt_builder.save("prompts/math_agent_base.json")
```

### 2. Run optimization

```bash
# Using the CLI
fair-optimize \
    --input prompts/math_agent_base.json \
    --output prompts/math_agent_optimized.json \
    --training-data training_examples.json \
    --optimizer mipro \
    --model cognitivecomputations/Dolphin3.0-Llama3.2-3B
```

### 3. Load optimized prompts in FAIR-LLM

```python
from fairlib.core.prompts import PromptBuilder

builder = PromptBuilder(optimized_path="prompts/math_agent_optimized.json")
# Now using optimized instructions and few-shot examples
```

## Optimizers

### BootstrapFewShot

Generates high-quality few-shot examples by running your program on training data and keeping examples that pass your metric.

**Best for:**
- Small datasets (10-50 examples)
- Quick optimization runs
- When you primarily need better examples, not instruction rewrites

```python
optimizer.optimize_bootstrap(
    fair_config_path="base.json",
    training_examples=data,
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

### MIPROv2

Jointly optimizes both instructions AND few-shot examples using Bayesian optimization.

**Best for:**
- Larger datasets (50+ examples)
- When you want instruction optimization
- Production-quality prompt tuning

```python
optimizer.optimize_mipro(
    fair_config_path="base.json",
    training_examples=data,
    metric=my_metric,
    auto="medium",  # "light", "medium", or "heavy"
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

## Training Data Format

Training examples can be provided as JSON:

```json
[
    {
        "inputs": {"user_query": "What is 2 + 2?"},
        "expected_output": "4"
    },
    {
        "inputs": {"user_query": "Calculate the square root of 16"},
        "expected_output": "4"
    }
]
```

## Metrics

Built-in metrics are available in `fair_prompt_optimizer.metrics`:

```python
from fair_prompt_optimizer.metrics import (
    exact_match,
    contains_answer,
    numeric_accuracy,
    semantic_similarity
)

# Use in optimization
optimizer.optimize_mipro(..., metric=exact_match)
```

## JSON Contract

The JSON format serves as the contract between FAIR-LLM and this optimizer:

```json
{
    "version": "1.0",
    "role_definition": "You are an expert mathematical calculator...",
    "tool_instructions": [
        {"name": "safe_calculator", "description": "Performs safe math operations"}
    ],
    "worker_instructions": [],
    "format_instructions": [
        "Your response must contain a 'Thought' and an 'Action' part."
    ],
    "examples": [
        "User: What is 2+2?\nThought: I should use the calculator.\nAction: {...}"
    ],
    "metadata": {
        "optimized": true,
        "optimized_at": "2025-11-24T10:30:00",
        "optimizer": "mipro",
        "score": 0.94
    }
}
```