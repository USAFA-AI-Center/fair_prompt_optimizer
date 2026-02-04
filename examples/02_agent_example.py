#!/usr/bin/env python3
"""
Example 2: Optimize a Single Agent

Use case: An agent with tools that follows the ReAct pattern
(Thought -> Action -> Observation cycles).

This example shows how to optimize an agent using full-trace examples
that demonstrate the complete tool-use workflow.

Usage:
    cd /home/ai-user/fair_prompt_optimizer
    source .venv/bin/activate
    python examples/02_agent_example.py
"""

# Suppress warnings from unneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Training data path
MATH_DATA_PATH = Path(__file__).parent / "training_data" / "math_examples.json"


async def main():
    """
    Optimize a math agent that uses a calculator tool.
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
        contains_answer,
        load_training_examples,
    )

    print("=" * 60)
    print("EXAMPLE: OPTIMIZE SINGLE AGENT")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Create agent with minimal prompts
    # -------------------------------------------------------------------------
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

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
    prompt_builder.role_definition = RoleDefinition("You are a math helper.")
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
    initial_path = Path(__file__).parent / "agent_initial.json"
    save_agent_config(agent, str(initial_path))
    print(f"Initial agent saved to: {initial_path}")
    print(f"Initial examples: 1 (seed example)")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Test baseline agent
    # -------------------------------------------------------------------------
    print("Testing baseline agent:")
    print("-" * 40)

    test_question = "What is 25 + 17?"
    agent.memory.clear()
    try:
        result = await agent.arun(test_question)
        print(f"Q: {test_question}")
        print(f"A: {result}")
    except Exception as e:
        print(f"Baseline test error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Load training examples and optimize
    # -------------------------------------------------------------------------
    if not MATH_DATA_PATH.exists():
        print(f"ERROR: Training data not found at {MATH_DATA_PATH}")
        return

    training_examples = load_training_examples(str(MATH_DATA_PATH))
    print(f"Loaded {len(training_examples)} training examples")
    print()

    optimizer = AgentOptimizer(agent)

    print("Starting optimization with Bootstrap...")
    print("-" * 40)

    result = optimizer.compile(
        training_examples=training_examples,
        metric=contains_answer,
        optimizer="bootstrap",
        max_bootstrapped_demos=3,
        training_data_path=str(MATH_DATA_PATH),
    )

    # -------------------------------------------------------------------------
    # Step 4: Save and show results
    # -------------------------------------------------------------------------
    output_path = Path(__file__).parent / "agent_optimized.json"
    result.save(str(output_path))

    print()
    print("=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)
    print(f"Optimized agent saved to: {output_path}")
    print(f"Examples: {len(result.examples)}")
    print()

    # Show what was optimized
    if result.optimization and result.optimization.runs:
        last_run = result.optimization.runs[-1]
        print(f"Optimizer: {last_run.optimizer}")
        print(f"Role definition changed: {last_run.role_definition_changed}")
        print(f"Format instructions changed: {last_run.format_instructions_changed}")
        print(f"Examples: {last_run.examples_before} -> {last_run.examples_after}")
        print()

        if last_run.role_definition_changed:
            print("New role_definition:")
            print("-" * 40)
            role_def = result.role_definition or ""
            print(role_def[:500] + "..." if len(role_def) > 500 else role_def)
            print()

    # -------------------------------------------------------------------------
    # Step 5: Test optimized agent
    # -------------------------------------------------------------------------
    print("Testing optimized agent:")
    print("-" * 40)

    optimized_agent = load_agent(str(output_path), llm)

    test_questions = [
        "What is 75 percent of 120?",
        "Calculate 256 / 16",
        "What is 33 + 67?",
    ]

    for q in test_questions:
        optimized_agent.memory.clear()
        try:
            result = await optimized_agent.arun(q)
            print(f"Q: {q}")
            print(f"A: {result}")
        except Exception as e:
            print(f"Q: {q}")
            print(f"Error: {e}")
        print()

    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
