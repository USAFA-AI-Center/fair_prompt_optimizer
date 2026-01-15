#!/usr/bin/env python3
"""
FAIR Prompt Optimizer - Complete Examples

Demonstrates all optimization workflows:
1. example_optimize_agent()        - Optimize a single agent with full-trace examples
2. example_optimize_simple_llm()   - Optimize a base LLM (no agent/tools)
3. example_optimize_multi_agent()  - Optimize a multi-agent system
4. example_iterative_optimization() - Multiple rounds of optimization
5. example_compare_agents()        - Compare optimized vs non-optimized
6. example_mipro_optimization()    - Use MIPROv2 for instruction optimization

All examples use:
- ReActPlanner (JSON format) for structured output
- Full-trace examples loaded from training_data/math_examples.json
"""

# suppress warnings from uneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
from pathlib import Path
import dspy

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

from fair_prompt_optimizer import load_training_examples

# paths to datasets
SENTIMENT_DATA_PATH = Path(__file__).parent / "training_data" / "sentiment_examples.json"
TRAINING_DATA_PATH = Path(__file__).parent / "training_data" / "math_examples.json"


# =============================================================================
# Example 1: Optimize a Single Agent
# =============================================================================

async def example_optimize_agent():
    """
    Optimize a single agent using AgentOptimizer with full-trace examples.
    
    This is the primary use case: take an agent, optimize its prompts with
    examples that show the complete tool-use workflow.
    """
    from fairlib import (
        HuggingFaceAdapter,
        SimpleAgent,
        ToolRegistry,
        ToolExecutor,
        WorkingMemory,
    )
    from fairlib.modules.planning.react_planner import ReActPlanner
    from fairlib.core.prompts import (
        PromptBuilder,
        RoleDefinition,
        FormatInstruction,
        Example
    )
    from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
    from fairlib.utils.config_manager import save_agent_config, load_agent
    
    from fair_prompt_optimizer import (
        AgentOptimizer,
        numeric_accuracy,
    )
    
    print("=" * 60)
    print("EXAMPLE 1: OPTIMIZE SINGLE AGENT")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Create agent with NO examples
    # -------------------------------------------------------------------------
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b") # TODO:: not great design here
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    
   # One seed example to show the model the correct format
    SEED_EXAMPLE = '''# --- Example ---
"What is 15 + 27?"

{
    "thought": "I need to add 15 and 27. I will use the calculator.",
    "action": {
        "tool_name": "safe_calculator",
        "tool_input": "15 + 27"
    }
}

Observation: The result of 15 + 27 is 42

{
    "thought": "The calculator returned 42. I will provide this as the final answer.",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "42"
    }
}'''
    
    prompt_builder = PromptBuilder()
    prompt_builder.role_definition = RoleDefinition(
        "You are a math helper."
    )
    prompt_builder.format_instructions = [
        FormatInstruction(
            "# --- RESPONSE FORMAT ---\n"
            "Respond with a JSON object containing 'thought' and 'action' keys.\n"
        ),
        FormatInstruction(
            "# --- FINAL ANSWER FORMAT ---\n"
            "When providing the final answer, use tool_name 'final_answer'"
        ),
    ]
    prompt_builder.examples = [Example(SEED_EXAMPLE)]
    
    planner = ReActPlanner(llm, tool_registry, prompt_builder=prompt_builder)
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=10,
    )
    
    # Save initial config
    save_agent_config(agent, "agent_initial.json")
    print(f"Initial agent: 0 examples")
    print(f"Saved to: agent_initial.json\n")

    
    # -------------------------------------------------------------------------
    # Step 2: Load training examples and optimize
    # -------------------------------------------------------------------------
    training_examples = load_training_examples(str(TRAINING_DATA_PATH))
    print(f"Loaded {len(training_examples)} training examples")
    
    optimizer = AgentOptimizer(agent)
    
    # needed for generating MIPRO candidate prompts
    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    result = optimizer.compile(
        training_examples=training_examples,
        metric=numeric_accuracy,
        optimizer="mipro",
        max_bootstrapped_demos=4,
        dspy_lm=dspy_lm,
        mipro_auto='light',
        training_data_path=str(TRAINING_DATA_PATH),
    )
    
    # Save optimized config
    result.save("agent_optimized.json")
    
    print()
    print(f"Optimized agent: {len(result.examples)} examples")
    print(f"Saved to: agent_optimized.json")
    print()
    
    # Show what was optimized
    last_run = result.optimization.runs[-1]
    print("Optimization results:")
    print(f"  - role_definition changed: {last_run.role_definition_changed}")
    print(f"  - format_instructions changed: {last_run.format_instructions_changed}")
    print()
    
    if last_run.role_definition_changed:
        print("New role_definition:")
        print("-" * 40)
        print(result.role_definition[:500] + "..." if len(result.role_definition or "") > 500 else result.role_definition)
        print()
    
    if last_run.format_instructions_changed:
        print(f"New format_instructions ({len(result.format_instructions)} items):")
        print("-" * 40)
        for i, fi in enumerate(result.format_instructions):
            preview = fi[:200] + "..." if len(fi) > 200 else fi
            print(f"  [{i}]: {preview}")
        print()
    
    # -------------------------------------------------------------------------
    # Step 3: Test optimized agent
    # -------------------------------------------------------------------------
    print("Testing optimized agent:")
    print("-" * 40)
    
    optimized_agent = load_agent("agent_optimized.json", llm)
    
    test_questions = ["What is 75 percent of 120?", "Calculate 256 / 16"]
    for q in test_questions:
        optimized_agent.memory.clear()
        result = await optimized_agent.arun(q)
        print(f"Q: {q}")
        print(f"A: {result}")
        print()
    
    return optimizer


# =============================================================================
# Example 2: Optimize a Simple LLM (No Agent)
# =============================================================================

async def example_optimize_simple_llm():
    """
    Optimize a base LLM with just a system prompt - no agent pipeline or tools.
    
    Use case: classification, format compliance, simple generation tasks.
    """
    from fairlib import HuggingFaceAdapter
    
    from fair_prompt_optimizer import (
        SimpleLLMOptimizer,
        format_compliance,
        load_optimized_config,
    )
    
    print("=" * 60)
    print("EXAMPLE 2: OPTIMIZE SIMPLE LLM (No Agent)")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # System prompt for sentiment classification
    system_prompt = """You are a sentiment classifier.

# --- RESPONSE FORMAT (MANDATORY) ---
Respond with ONLY one of:
SENTIMENT: positive
SENTIMENT: negative
SENTIMENT: neutral

No explanations, just the label."""

    # Training examples (no full_trace needed for simple LLM)
    training_examples = load_training_examples(str(SENTIMENT_DATA_PATH))
    
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Training examples: {len(training_examples)}")
    print()

    opt_config = load_optimized_config("classifier_initial.json")
    
    optimizer = SimpleLLMOptimizer(llm=llm, system_prompt=system_prompt, config=opt_config)
    
    # needed for generating MIPRO candidate prompts
    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    result = optimizer.compile(
        training_examples=training_examples,
        metric=format_compliance("SENTIMENT:"),
        optimizer="bootstrap",
        # dspy_lm=dspy_lm,
        max_bootstrapped_demos=3,
    )
    
    result.save("classifier_optimized.json")
    
    print(f"Optimized examples: {len(result.prompts.get('examples', []))}")
    print(f"Saved to: classifier_optimized.json")
    print()
    
    # Test
    print("Testing optimized classifier:")
    print("-" * 40)
    for text in ["Amazing!", "Horrible experience.", "Its okay."]:
        response = optimizer.test(text)
        print(f"Input: {text}")
        print(f"Output: {response}")
        print()
    
    return optimizer


# =============================================================================
# Example 3: Optimize a Multi-Agent System
# =============================================================================

async def example_optimize_multi_agent():
    """
    Optimize a multi-agent system with manager and workers.
    
    Use case: complex tasks requiring coordination between specialized agents.
    """
    from fairlib import (
        HuggingFaceAdapter,
        SimpleAgent,
        ToolRegistry,
        ToolExecutor,
        WorkingMemory,
    )
    from fairlib.modules.planning.react_planner import ReActPlanner
    from fairlib.core.prompts import (
        PromptBuilder,
        RoleDefinition,
        FormatInstruction,
        WorkerInstruction,
    )
    from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
    from fairlib.modules.orchestration.hierarchical_runner import HierarchicalAgentRunner
    from fairlib.utils.config_manager import save_multi_agent_config, load_multi_agent
    
    from fair_prompt_optimizer import (
        MultiAgentOptimizer,
        load_training_examples,
        contains_answer,
    )
    
    print("=" * 60)
    print("EXAMPLE 3: OPTIMIZE MULTI-AGENT SYSTEM")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # -------------------------------------------------------------------------
    # Create Calculator Worker
    # -------------------------------------------------------------------------
    calc_registry = ToolRegistry()
    calc_registry.register_tool(SafeCalculatorTool())
    
    calc_builder = PromptBuilder()
    calc_builder.role_definition = RoleDefinition(
        "You are a calculator specialist. Use safe_calculator for all math."
    )
    calc_builder.format_instructions = [
        FormatInstruction(
            "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"...\", \"tool_input\": \"...\"}}"
        ),
    ]
    calc_builder.examples = []
    
    calc_planner = ReActPlanner(llm, calc_registry, prompt_builder=calc_builder)
    calculator_worker = SimpleAgent(
        llm=llm,
        planner=calc_planner,
        tool_executor=ToolExecutor(calc_registry),
        memory=WorkingMemory(),
        max_steps=5,
    )
    
    # -------------------------------------------------------------------------
    # Create Manager
    # -------------------------------------------------------------------------
    manager_registry = ToolRegistry()
    
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are a manager that delegates math problems to the Calculator worker."
    )
    manager_builder.format_instructions = [
        FormatInstruction(
            "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"Calculator\", \"tool_input\": \"task\"}}"
        ),
    ]
    manager_builder.worker_instructions = [
        WorkerInstruction(name="Calculator", role_description="Handles math calculations."),
    ]
    manager_builder.examples = []
    
    manager_planner = ReActPlanner(llm, manager_registry, prompt_builder=manager_builder)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=ToolExecutor(manager_registry),
        memory=WorkingMemory(),
        max_steps=5,
    )
    
    # -------------------------------------------------------------------------
    # Create Runner and Optimize
    # -------------------------------------------------------------------------
    runner = HierarchicalAgentRunner(
        manager=manager_agent,
        workers={"Calculator": calculator_worker},
    )
    
    save_multi_agent_config(runner, "multi_agent_initial.json")
    print(f"Initial multi-agent system saved")
    
    training_examples = load_training_examples(str(TRAINING_DATA_PATH))
    
    optimizer = MultiAgentOptimizer(runner)
    
    result = optimizer.compile(
        training_examples=training_examples[:4],
        metric=contains_answer,
        optimizer="bootstrap",
        optimize_manager=True,
        optimize_workers=False,
        max_bootstrapped_demos=2,
    )
    
    result.save("multi_agent_optimized.json")
    
    print(f"Optimized multi-agent system saved")
    print(f"Optimization runs: {len(result.optimization.runs)}")
    print()
    
    # Test
    print("Testing optimized multi-agent system:")
    print("-" * 40)
    
    optimized_runner = load_multi_agent("multi_agent_optimized.json", llm)
    test_result = await optimized_runner.run("What is 50% of 80?")
    print(f"Q: What is 50% of 80?")
    print(f"A: {test_result}")
    
    return optimizer


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("FAIR PROMPT OPTIMIZER - EXAMPLES")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Uncomment the example(s) you want to run
    # -------------------------------------------------------------------------
    
    # Example 1: Optimize a single agent
    print("Running: example_optimize_agent")
    asyncio.run(example_optimize_agent())
    
    # Example 2: Optimize a simple LLM (no agent)
    # print("Running: example_optimize_simple_llm")
    # asyncio.run(example_optimize_simple_llm())
    
    # Example 3: Optimize a multi-agent system
    # print("Running: example_optimize_multi_agent")
    # asyncio.run(example_optimize_multi_agent())