#registry.py

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
    """Register a custom adapter class."""
    ADAPTER_REGISTRY[name] = cls


def register_tool(name: str, cls: Type):
    """Register a custom tool class."""
    TOOL_REGISTRY[name] = cls


def register_planner(name: str, cls: Type):
    """Register a custom planner class."""
    PLANNER_REGISTRY[name] = cls


def register_agent(name: str, cls: Type):
    """Register a custom agent class."""
    AGENT_REGISTRY[name] = cls


# =============================================================================
# Agent Loading
# =============================================================================

def load_optimized_agent(config_path: str):
    """
    Create a fully configured FAIR-LLM agent from an optimized config file.
    
    Args:
        config_path: Path to the optimized config JSON file
        
    Returns:
        Ready-to-use FAIR-LLM agent with optimized prompts
        
    Example:
        from fair_prompt_optimizer import load_optimized_agent
        
        agent = load_optimized_agent("prompts/optimized.json")
        response = await agent.arun("What is 15 + 27?")
    """
    # Ensure registries are populated
    _populate_registries()
    
    # Import fairlib components we need directly
    from fairlib import ToolRegistry, ToolExecutor, WorkingMemory, RoleDefinition
    
    # Load config
    config = load_fair_config(config_path)
    
    # --- Create LLM ---
    adapter_name = config.model.adapter
    if adapter_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter: {adapter_name}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    
    adapter_class = ADAPTER_REGISTRY[adapter_name]
    llm = adapter_class(config.model.model_name, **config.model.adapter_kwargs)
    logger.info(f"Created LLM: {adapter_name}({config.model.model_name})")
    
    # --- Create Tool Registry ---
    tool_registry = ToolRegistry()
    for tool_name in config.agent.tools:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(
                f"Unknown tool: {tool_name}. "
                f"Available: {list(TOOL_REGISTRY.keys())}"
            )
        tool_class = TOOL_REGISTRY[tool_name]
        tool_registry.register_tool(tool_class())
        logger.debug(f"Registered tool: {tool_name}")
    
    # --- Create Planner with Optimized Prompt ---
    planner_name = config.agent.planner_type
    if planner_name not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner: {planner_name}. "
            f"Available: {list(PLANNER_REGISTRY.keys())}"
        )
    
    planner_class = PLANNER_REGISTRY[planner_name]
    planner = planner_class(llm, tool_registry)
    
    # Apply optimized role definition
    if config.role_definition:
        planner.prompt_builder.role_definition = RoleDefinition(config.role_definition)
        logger.debug("Applied optimized role_definition")
    
    # Apply optimized examples if supported
    if config.examples and hasattr(planner.prompt_builder, 'examples'):
        planner.prompt_builder.examples = config.examples
        logger.debug(f"Applied {len(config.examples)} optimized examples")
    
    # --- Create Agent ---
    agent_name = config.agent.agent_type
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_name}. "
            f"Available: {list(AGENT_REGISTRY.keys())}"
        )
    
    agent_class = AGENT_REGISTRY[agent_name]
    agent = agent_class(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=config.agent.max_steps
    )
    
    logger.info(
        f"Created optimized agent: {agent_name} with {len(config.agent.tools)} tools"
    )
    
    return agent


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
    
    if config.examples and hasattr(planner.prompt_builder, 'examples'):
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