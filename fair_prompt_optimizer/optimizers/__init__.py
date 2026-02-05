# fair_prompt_optimizer/optimizers/__init__.py
"""
DSPy-compatible modules and optimizers for FAIR-LLM.

Three optimization levels:
1. SimpleLLMOptimizer - LLM with system prompt (no agent pipeline)
2. AgentOptimizer - Single SimpleAgent with tools
3. MultiAgentOptimizer - HierarchicalAgentRunner with manager + workers

All modules follow the FAIRAgentModule pattern:
- Create a DSPy signature from the component's config
- Create a dspy.Predict for DSPy to attach demos to
- forward() runs the real component and returns dspy.Prediction
- get_config() extracts optimized demos from self.predict.demos

This package re-exports all public symbols for backward compatibility.
Import from here or from individual submodules.
"""

# Base utilities
# Level 2: Single Agent
from .agent import (
    AgentModule,
    AgentOptimizer,
)
from .base import (
    FORMAT_END,
    FORMAT_ITEM_END,
    FORMAT_ITEM_START,
    FORMAT_START,
    ROLE_END,
    ROLE_START,
    clear_cuda_memory,
    combine_prompt_components,
    parse_optimized_prompt,
    run_async,
)

# Level 3: Multi-Agent
from .multi_agent import (
    MultiAgentModule,
    MultiAgentOptimizer,
)

# Level 1: Simple LLM
from .simple_llm import (
    SimpleLLMModule,
    SimpleLLMOptimizer,
)

__all__ = [
    # Base utilities
    "ROLE_START",
    "ROLE_END",
    "FORMAT_START",
    "FORMAT_END",
    "FORMAT_ITEM_START",
    "FORMAT_ITEM_END",
    "combine_prompt_components",
    "parse_optimized_prompt",
    "run_async",
    "clear_cuda_memory",
    # Level 1: Simple LLM
    "SimpleLLMModule",
    "SimpleLLMOptimizer",
    # Level 2: Single Agent
    "AgentModule",
    "AgentOptimizer",
    # Level 3: Multi-Agent
    "MultiAgentModule",
    "MultiAgentOptimizer",
]
