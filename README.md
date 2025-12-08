# fair_prompt_optimizer

DSPy-powered prompt optimization for FAIR-LLM agents.

## Overview

This tool optimizes FAIR-LLM agent prompts by running **actual agents** (with tools, memory, etc.) on training examples and using DSPy's optimization algorithms to improve performance.

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

## Quick Start

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
from fair_prompt_optimizer import FAIRPromptOptimizer, TrainingExample, numeric_accuracy

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

# 3. Optimize!
optimizer = FAIRPromptOptimizer(agent)
config = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="bootstrap",  # or "mipro"
    output_path="optimized_config.json"
)

# 4. Apply optimized config back to agent
optimizer.apply_to_agent(config)
```

## Example

See `examples/optimize_fair_agent.py` for a complete working example.