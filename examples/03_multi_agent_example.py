#!/usr/bin/env python3
"""
Example 3: Optimize a Multi-Agent System

Use case: A hierarchical agent system with a manager and multiple workers.

This example shows how to optimize a research team:
- Manager: Orchestrates research tasks
- DataGatherer: Searches for information
- Summarizer: Synthesizes and summarizes findings

Usage:
    cd /home/ai-user/fair_prompt_optimizer
    source .venv/bin/activate
    python examples/03_multi_agent_example.py
"""

# Suppress warnings from unneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
import os
from pathlib import Path
import dspy

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Training data paths
RESEARCH_MANAGER_DATA_PATH = Path(__file__).parent / "training_data" / "research_manager_examples.json"
RESEARCH_DATAGATHERER_DATA_PATH = Path(__file__).parent / "training_data" / "research_datagatherer_examples.json"
RESEARCH_SUMMARIZER_DATA_PATH = Path(__file__).parent / "training_data" / "research_summarizer_examples.json"


async def main():
    """
    Optimize a research team multi-agent system.
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
    from fairlib.modules.agent.multi_agent_runner import HierarchicalAgentRunner
    from fairlib.utils.config_manager import save_multi_agent_config, load_multi_agent

    from fair_prompt_optimizer import (
        MultiAgentOptimizer,
        load_training_examples,
        research_quality_metric,
    )

    print("=" * 60)
    print("EXAMPLE: OPTIMIZE MULTI-AGENT RESEARCH TEAM")
    print("=" * 60)
    print()

    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # -------------------------------------------------------------------------
    # Create DataGatherer Worker
    # -------------------------------------------------------------------------
    gatherer_registry = ToolRegistry()
    # Note: WebSearcherTool requires API keys. For this example, we use no tools.
    # In production, you would register: gatherer_registry.register_tool(WebSearcherTool(config))

    gatherer_builder = PromptBuilder()
    gatherer_builder.role_definition = RoleDefinition(
        "You are a research data gatherer. Find accurate, relevant information on the given topic."
    )
    gatherer_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
        ),
        FormatInstruction(
            '# --- FINAL ANSWER FORMAT ---\n'
            'Use tool_name "final_answer" with the gathered information as tool_input.'
        ),
    ]

    gatherer_planner = ReActPlanner(llm, gatherer_registry, prompt_builder=gatherer_builder)
    data_gatherer = SimpleAgent(
        llm=llm,
        planner=gatherer_planner,
        tool_executor=ToolExecutor(gatherer_registry),
        memory=WorkingMemory(),
        max_steps=3,
    )

    # -------------------------------------------------------------------------
    # Create Summarizer Worker
    # -------------------------------------------------------------------------
    summarizer_registry = ToolRegistry()  # No tools - pure LLM reasoning

    summarizer_builder = PromptBuilder()
    summarizer_builder.role_definition = RoleDefinition(
        "You are a research summarizer. Synthesize raw data into clear, concise summaries."
    )
    summarizer_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
        ),
        FormatInstruction(
            '# --- FINAL ANSWER FORMAT ---\n'
            'Provide your summary using tool_name "final_answer" with the summary as tool_input.'
        ),
    ]

    summarizer_planner = ReActPlanner(llm, summarizer_registry, prompt_builder=summarizer_builder)
    summarizer = SimpleAgent(
        llm=llm,
        planner=summarizer_planner,
        tool_executor=ToolExecutor(summarizer_registry),
        memory=WorkingMemory(),
        max_steps=3,
    )

    # -------------------------------------------------------------------------
    # Create Manager Agent
    # -------------------------------------------------------------------------
    manager_registry = ToolRegistry()  # Workers are registered by HierarchicalAgentRunner

    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are a research team manager. Coordinate research tasks by delegating to "
        "DataGatherer for information collection, then Summarizer for synthesis."
    )
    manager_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
            'Delegate tasks to workers using their names as tool_name.\n'
        ),
        FormatInstruction(
            '# --- WORKFLOW ---\n'
            '1. Delegate to DataGatherer to gather information\n'
            '2. Delegate to Summarizer with the gathered data\n'
            '3. Provide the final answer using "final_answer"'
        ),
    ]
    manager_builder.worker_instructions = [
        WorkerInstruction(
            name="DataGatherer",
            role_description="Searches for relevant information. Input: search topic. Output: raw findings."
        ),
        WorkerInstruction(
            name="Summarizer",
            role_description="Synthesizes raw data into clear summaries. Input: raw data. Output: concise summary."
        ),
    ]

    manager_planner = ReActPlanner(llm, manager_registry, prompt_builder=manager_builder)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=ToolExecutor(manager_registry),
        memory=WorkingMemory(),
        max_steps=8,
    )

    # -------------------------------------------------------------------------
    # Create Hierarchical Runner
    # -------------------------------------------------------------------------
    runner = HierarchicalAgentRunner(
        manager_agent=manager_agent,
        workers={
            "DataGatherer": data_gatherer,
            "Summarizer": summarizer,
        },
    )

    # Save initial config
    initial_path = Path(__file__).parent / "research_team_initial.json"
    save_multi_agent_config(runner, str(initial_path))
    print(f"Initial research team saved to: {initial_path}")
    print()

    # -------------------------------------------------------------------------
    # Test baseline (before optimization)
    # -------------------------------------------------------------------------
    print("Testing baseline research team:")
    print("-" * 40)

    test_query = "What are the benefits of renewable energy?"
    try:
        result = await runner.arun(test_query)
        print(f"Q: {test_query}")
        result_str = str(result)
        print(f"A: {result_str[:500]}..." if len(result_str) > 500 else f"A: {result}")
    except Exception as e:
        print(f"Baseline test error: {e}")
    print()

    # -------------------------------------------------------------------------
    # Load training examples
    # -------------------------------------------------------------------------
    if not RESEARCH_MANAGER_DATA_PATH.exists():
        print(f"ERROR: Manager training data not found at {RESEARCH_MANAGER_DATA_PATH}")
        return

    manager_examples = load_training_examples(str(RESEARCH_MANAGER_DATA_PATH))
    print(f"Loaded {len(manager_examples)} manager training examples")

    # Load worker training examples
    worker_training_examples = {}

    if RESEARCH_DATAGATHERER_DATA_PATH.exists():
        gatherer_examples = load_training_examples(str(RESEARCH_DATAGATHERER_DATA_PATH))
        worker_training_examples["DataGatherer"] = gatherer_examples
        print(f"Loaded {len(gatherer_examples)} DataGatherer training examples")

    if RESEARCH_SUMMARIZER_DATA_PATH.exists():
        summarizer_examples = load_training_examples(str(RESEARCH_SUMMARIZER_DATA_PATH))
        worker_training_examples["Summarizer"] = summarizer_examples
        print(f"Loaded {len(summarizer_examples)} Summarizer training examples")

    print()

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    optimizer = MultiAgentOptimizer(
        runner,
        optimize_manager=True,
        optimize_workers=True,  # Enable recursive worker optimization
    )

    # DSPy LM for generating MIPRO candidate prompts
    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    print("Starting multi-agent optimization with Bootstrap...")
    print("-" * 40)

    result = optimizer.compile(
        training_examples=manager_examples,
        metric=research_quality_metric,
        worker_training_examples=worker_training_examples,
        optimizer="bootstrap",  # Use bootstrap for faster iteration
        max_bootstrapped_demos=2,
        dspy_lm=dspy_lm,
    )

    # -------------------------------------------------------------------------
    # Save and show results
    # -------------------------------------------------------------------------
    output_path = Path(__file__).parent / "research_team_optimized.json"
    result.save(str(output_path))

    print()
    print("=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)
    print(f"Config saved to: {output_path}")
    print()

    # Show what was optimized
    if result.optimization and result.optimization.runs:
        last_run = result.optimization.runs[-1]
        print(f"Optimizer: {last_run.optimizer}")
        print(f"Manager role_definition changed: {last_run.role_definition_changed}")
        print(f"Manager examples: {last_run.examples_before} -> {last_run.examples_after}")

        workers_optimized = last_run.optimizer_config.get("workers_optimized", [])
        if workers_optimized:
            print(f"Workers optimized: {workers_optimized}")
    print()

    # -------------------------------------------------------------------------
    # Test optimized research team
    # -------------------------------------------------------------------------
    print("Testing optimized research team:")
    print("-" * 40)

    # Reload from saved config
    optimized_runner = load_multi_agent(str(output_path), llm)

    test_queries = [
        "What are the main causes of climate change?",
        "Summarize the history of artificial intelligence.",
    ]

    for query in test_queries:
        try:
            result = await optimized_runner.arun(query)
            print(f"Q: {query}")
            result_str = str(result)
            print(f"A: {result_str[:300]}..." if len(result_str) > 300 else f"A: {result}")
        except Exception as e:
            print(f"Q: {query}")
            print(f"Error: {e}")
        print()

    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
