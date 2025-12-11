# registry.py
"""
Registry pattern for instantiating FAIR-LLM components from config.

This file provides a way to create agents from config files without
hardcoding class names. When moved to fair_llm, these registries can
be auto-populated or extended by users.

Usage:
    from fair_prompt_optimizer.registry import load_optimized_agent
    
    agent = load_optimized_agent("prompts/optimized.json")
"""

import logging
from typing import Dict, Type

from .translator import FAIRConfig, load_fair_config

logger = logging.getLogger(__name__)


# =============================================================================
# Component Registries
# =============================================================================

ADAPTER_REGISTRY: Dict[str, Type] = {}
TOOL_REGISTRY: Dict[str, Type] = {}
PLANNER_REGISTRY: Dict[str, Type] = {}
AGENT_REGISTRY: Dict[str, Type] = {}


def _populate_registries():
    """
    Populate registries with FAIR-LLM classes.
    
    Called lazily on first use to avoid import errors if fairlib not installed.
    """
    global ADAPTER_REGISTRY, TOOL_REGISTRY, PLANNER_REGISTRY, AGENT_REGISTRY
    
    if ADAPTER_REGISTRY:  # Already populated
        return
    
    try:
        from fairlib import (
            # Adapters
            HuggingFaceAdapter,
            OpenAIAdapter,
            AnthropicAdapter,
            # Tools
            SafeCalculatorTool,
            # Planners
            SimpleReActPlanner,
            # Agents
            SimpleAgent,
            # Components
            ToolRegistry,
            ToolExecutor,
            WorkingMemory,
            RoleDefinition,
        )
        
        ADAPTER_REGISTRY.update({
            "HuggingFaceAdapter": HuggingFaceAdapter,
            "OpenAIAdapter": OpenAIAdapter,
            "AnthropicAdapter": AnthropicAdapter,
            # Add more adapters as created
        })
        
        TOOL_REGISTRY.update({
            "SafeCalculatorTool": SafeCalculatorTool,
            # Add more tools here
        })
        
        PLANNER_REGISTRY.update({
            "SimpleReActPlanner": SimpleReActPlanner,
            # Add more planners as created
        })
        
        AGENT_REGISTRY.update({
            "SimpleAgent": SimpleAgent,
            # Add more agent types as created
        })
        
        logger.debug("FAIR-LLM registries populated successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import fairlib components: {e}")
        raise ImportError(
            "fairlib is required to load optimized agents. "
            "Install it with: pip install fair-llm"
        )


def register_adapter(name: str, cls: Type):
    ADAPTER_REGISTRY[name] = cls


def register_tool(name: str, cls: Type):
    TOOL_REGISTRY[name] = cls


def register_planner(name: str, cls: Type):
    PLANNER_REGISTRY[name] = cls


def register_agent(name: str, cls: Type):
    AGENT_REGISTRY[name] = cls


# =============================================================================
# Agent Loading
# =============================================================================


def create_agent_from_config(config: FAIRConfig):
    """
    Create a FAIR-LLM agent from a FAIRConfig object.
    
    Same as load_optimized_agent but takes a config object instead of path.
    
    Args:
        config: FAIRConfig object
        
    Returns:
        Ready-to-use FAIR-LLM agent
    """
    # Ensure registries are populated
    _populate_registries()
    
    from fairlib import ToolRegistry, ToolExecutor, WorkingMemory, RoleDefinition
    
    # Create LLM
    adapter_class = ADAPTER_REGISTRY[config.model.adapter]
    llm = adapter_class(config.model.model_name, **config.model.adapter_kwargs)
    
    # Create Tool Registry
    tool_registry = ToolRegistry()
    for tool_name in config.agent.tools:
        tool_class = TOOL_REGISTRY[tool_name]
        tool_registry.register_tool(tool_class())
    
    # Create Planner
    planner_class = PLANNER_REGISTRY[config.agent.planner_type]
    planner = planner_class(llm, tool_registry)
    
    if config.role_definition:
        planner.prompt_builder.role_definition = RoleDefinition(config.role_definition)
    
    if config.examples:
        planner.prompt_builder.examples = config.examples
    
    # Create Agent
    agent_class = AGENT_REGISTRY[config.agent.agent_type]
    agent = agent_class(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=config.agent.max_steps
    )
    
    return agent

# TODO:: this method calls 'load_fair_config()' this lives in translator.py, if this registry is moved to fair_llm, places a dependency on this repository
# TODO:: May need to also implement load_fair_config in fair_llm
def load_optimized_agent(config_path: str):
    """
    Create a fully configured FAIR-LLM agent from an optimized config file.
    
    Args:
        config_path: Path to the optimized config JSON file
        
    Returns:
        Ready-to-use FAIR-LLM agent with optimized prompts
    """
    config = load_fair_config(config_path)
    return create_agent_from_config(config)