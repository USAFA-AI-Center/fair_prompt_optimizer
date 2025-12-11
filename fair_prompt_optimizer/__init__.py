"""
fair_prompt_optimizer - DSPy-powered prompt optimization for FAIR-LLM agents
"""

__version__ = "0.1.0"

from .translator import (
    TrainingExample,
    FAIRConfig,
    ToolInstruction,
    WorkerInstruction,
    OptimizationMetadata,
    ModelConfig,
    AgentConfig,
    DSPyTranslator,
    load_fair_config,
    save_fair_config,
    load_training_examples,
    save_training_examples,
)

from .metrics import (
    exact_match,
    contains_answer,
    numeric_accuracy,
    fuzzy_match,
)

from .fair_agent_module import (
    FAIRPromptOptimizer,
    FAIRAgentModule,
)

from .registry import (
    load_optimized_agent,
    create_agent_from_config,
    register_adapter,
    register_tool,
    register_planner,
    register_agent,
)

__all__ = [
    "__version__",
    # Core data structures
    "TrainingExample",
    "FAIRConfig",
    "ToolInstruction", 
    "WorkerInstruction",
    "OptimizationMetadata",
    "ModelConfig",
    "AgentConfig",
    "DSPyTranslator",
    # Config I/O
    "load_fair_config",
    "save_fair_config",
    "load_training_examples",
    "save_training_examples",
    # Metrics
    "exact_match",
    "contains_answer",
    "numeric_accuracy",
    "fuzzy_match",
    # Optimizer
    "FAIRPromptOptimizer",
    "FAIRAgentModule",
    # Agent Loading (registry)
    "load_optimized_agent",
    "create_agent_from_config",
    "register_adapter",
    "register_tool",
    "register_planner",
    "register_agent",
]
