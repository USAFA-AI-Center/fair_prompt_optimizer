# fair_prompt_optimizer/__init__.py
"""
FAIR Prompt Optimizer - DSPy-powered optimization for FAIR-LLM agents.

This library integrates with fairlib to provide prompt optimization using DSPy.
Configs are shared between both libraries for seamless workflow.

Three Levels of Optimization:
-----------------------------

Level 1: SimpleLLMOptimizer
    Optimize raw LLM + system prompt (no agent pipeline).
    Good for: classification, format compliance, simple generation.

    optimizer = SimpleLLMOptimizer(llm, "You are a classifier...")
    result = optimizer.compile(examples, metric)

Level 2: AgentOptimizer
    Optimize a fairlib SimpleAgent with tools and ReAct.
    Good for: tool-using agents, multi-step reasoning.

    from fairlib.utils.config_manager import load_agent
    agent = load_agent("config.json", llm)
    optimizer = AgentOptimizer(agent)
    result = optimizer.compile(examples, metric)

Level 3: MultiAgentOptimizer
    Optimize a fairlib HierarchicalAgentRunner (manager + workers).
    Good for: complex tasks requiring specialization.

    from fairlib.utils.config_manager import load_multi_agent
    runner = load_multi_agent("config.json", llm)
    optimizer = MultiAgentOptimizer(runner)
    result = optimizer.compile(examples, metric)

NOTE: This package does NOT re-export fairlib functions. Import them directly:
    from fairlib.utils.config_manager import load_agent, save_agent_config
"""

__version__ = "0.0.0"

# =============================================================================
# Config (our additions to fairlib's config system)
# =============================================================================
from .config import (
    # DSPy translation
    DSPyTranslator,
    # Optimization provenance
    OptimizationProvenance,
    OptimizationRun,
    # Config wrapper
    OptimizedConfig,
    # Training data
    TrainingExample,
    compute_file_hash,
    # I/O
    load_optimized_config,
    load_training_examples,
    save_optimized_config,
    save_training_examples,
)

# =============================================================================
# Metrics
# =============================================================================
from .metrics import (
    combined_metric,
    contains_answer,
    create_metric,
    exact_match,
    format_compliance,
    format_compliance_score,
    fuzzy_match,
    json_format_compliance,
    keyword_match,
    numeric_accuracy,
    numeric_accuracy_with_format,
    research_quality_metric,
    sentiment_format_metric,
)

# =============================================================================
# Optimizers
# =============================================================================
from .optimizers import (
    AgentModule,
    # Level 2: Single Agent
    AgentOptimizer,
    MultiAgentModule,
    # Level 3: Multi-Agent
    MultiAgentOptimizer,
    SimpleLLMModule,
    # Level 1: Simple LLM
    SimpleLLMOptimizer,
    clear_cuda_memory,
    # Utilities
    run_async,
)
