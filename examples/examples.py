#!/usr/bin/env python3
"""
FAIR Prompt Optimizer - Complete Examples
"""

# suppress warnings from uneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
import os
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
RESEARCH_MANAGER_DATA_PATH = Path(__file__).parent / "training_data" / "research_manager_examples.json"
RESEARCH_DATAGATHERER_DATA_PATH = Path(__file__).parent / "training_data" / "research_datagatherer_examples.json"
RESEARCH_SUMMARIZER_DATA_PATH = Path(__file__).parent / "training_data" / "research_summarizer_examples.json"


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
        sentiment_format_metric,
        load_optimized_config,
        format_compliance,
    )
    
    print("=" * 60)
    print("EXAMPLE 2: OPTIMIZE SIMPLE LLM (No Agent)")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # Training examples (no full_trace needed for simple LLM)
    training_examples = load_training_examples(str(SENTIMENT_DATA_PATH))

    opt_config = load_optimized_config("classifier_initial.json")
    
    optimizer = SimpleLLMOptimizer(llm=llm, system_prompt=opt_config.prompts["system_prompt"], config=opt_config)
    
    # needed for generating MIPRO candidate prompts
    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    result = optimizer.compile(
        training_examples=training_examples,
        metric=format_compliance("SENTIMENT:"),
        optimizer="mipro",
        mipro_auto="medium",
        dspy_lm=dspy_lm,
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

async def example_optimize_research_team():
    """
    Optimize a research team multi-agent system.
    
    Architecture:
    - Manager: Orchestrates research tasks
    - DataGatherer: Searches the web for information
    - Summarizer: Synthesizes and summarizes findings
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
        Example,
    )
    from fairlib.modules.orchestration.hierarchical_runner import HierarchicalAgentRunner
    from fairlib.utils.config_manager import save_multi_agent_config, load_multi_agent
    from fairlib.modules.action.tools.web_searcher import WebSearcherTool
    
    from fair_prompt_optimizer import (
        MultiAgentOptimizer,
        load_training_examples,
    )
    
    print("=" * 60)
    print("EXAMPLE 4: OPTIMIZE RESEARCH TEAM")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # -------------------------------------------------------------------------
    # Web Search Configuration
    # -------------------------------------------------------------------------
    web_search_config = {
        "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "google_search_engine_id": os.environ.get("GOOGLE_SEARCH_ENGINE_ID", ""),
        "max_results": 5,
        "cache_ttl": 3600,
    }
    
    # -------------------------------------------------------------------------
    # Create DataGatherer Worker
    # -------------------------------------------------------------------------
    gatherer_registry = ToolRegistry()
    gatherer_registry.register_tool(WebSearcherTool(web_search_config))
    
    gatherer_builder = PromptBuilder()
    gatherer_builder.role_definition = RoleDefinition(
        "You are a research data gatherer. Your job is to search the web for "
        "relevant, accurate, and up-to-date information on the given topic. "
        "Focus on finding authoritative sources."
    )
    gatherer_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
            'Use the web_searcher tool to find information.\n'
            'Example: {"thought": "I need to search for...", "action": {"tool_name": "web_searcher", "tool_input": "search query"}}'
        ),
        FormatInstruction(
            '# --- FINAL ANSWER FORMAT ---\n'
            'When you have gathered sufficient information, use tool_name "final_answer" '
            'with the search results as tool_input.'
        ),
    ]
    
    gatherer_planner = ReActPlanner(llm, gatherer_registry, prompt_builder=gatherer_builder)
    data_gatherer = SimpleAgent(
        llm=llm,
        planner=gatherer_planner,
        tool_executor=ToolExecutor(gatherer_registry),
        memory=WorkingMemory(),
        max_steps=5,
    )
    
    # -------------------------------------------------------------------------
    # Create Summarizer Worker
    # -------------------------------------------------------------------------
    summarizer_registry = ToolRegistry()  # No tools - pure LLM reasoning
    
    summarizer_builder = PromptBuilder()
    summarizer_builder.role_definition = RoleDefinition(
        "You are a research summarizer. Your job is to take raw research data "
        "and synthesize it into a clear, concise, and accurate summary. "
        "Highlight key findings, identify patterns, and note any conflicting information."
    )
    summarizer_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
            'Analyze the provided research data and formulate your summary.'
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
    manager_registry = ToolRegistry()  # Workers are registered as tools by HierarchicalAgentRunner
    
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are a research team manager. Your job is to coordinate research tasks "
        "by delegating to your team members. First use DataGatherer to collect information, "
        "then use Summarizer to synthesize the findings into a coherent response."
    )
    manager_builder.format_instructions = [
        FormatInstruction(
            '# --- RESPONSE FORMAT ---\n'
            'Respond with a JSON object containing "thought" and "action" keys.\n'
            'Delegate tasks to your workers using their names as tool_name.\n'
            'Example: {"thought": "I need to gather data on...", "action": {"tool_name": "DataGatherer", "tool_input": "research topic"}}'
        ),
        FormatInstruction(
            '# --- WORKFLOW ---\n'
            '1. First, delegate to DataGatherer to search for information\n'
            '2. Then, delegate to Summarizer with the gathered data\n'
            '3. Finally, provide the summarized response as final_answer'
        ),
    ]
    manager_builder.worker_instructions = [
        WorkerInstruction(
            name="DataGatherer", 
            role_description="Searches the web for relevant information on a topic. Input: search topic. Output: raw search results."
        ),
        WorkerInstruction(
            name="Summarizer", 
            role_description="Synthesizes raw data into a clear summary. Input: raw research data. Output: concise summary."
        ),
    ]
    
    manager_planner = ReActPlanner(llm, manager_registry, prompt_builder=manager_builder)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=ToolExecutor(manager_registry),
        memory=WorkingMemory(),
        max_steps=8,  # More steps for multi-hop coordination
    )
    
    # -------------------------------------------------------------------------
    # Create Runner
    # -------------------------------------------------------------------------
    runner = HierarchicalAgentRunner(
        manager=manager_agent,
        workers={
            "DataGatherer": data_gatherer,
            "Summarizer": summarizer,
        },
    )
    
    save_multi_agent_config(runner, "research_team_initial.json")
    print("Initial research team saved")
    
    # -------------------------------------------------------------------------
    # Test baseline (before optimization)
    # -------------------------------------------------------------------------
    print("\nTesting baseline research team:")
    print("-" * 40)
    
    test_query = "What are the latest developments in quantum computing?"
    try:
        result = await runner.run(test_query)
        print(f"Q: {test_query}")
        print(f"A: {result[:500]}..." if len(str(result)) > 500 else f"A: {result}")
    except Exception as e:
        print(f"Baseline test failed: {e}")
    
    # -------------------------------------------------------------------------
    # Load training examples and optimize
    # -------------------------------------------------------------------------

    # Load manager training examples
    if not RESEARCH_MANAGER_DATA_PATH.exists():
        print(f"\nNo manager training data found at {RESEARCH_MANAGER_DATA_PATH}")
        return runner

    manager_examples = load_training_examples(str(RESEARCH_MANAGER_DATA_PATH))
    print(f"\nLoaded {len(manager_examples)} manager training examples")

    # Load worker training examples (for recursive optimization)
    worker_training_examples = {}

    if RESEARCH_DATAGATHERER_DATA_PATH.exists():
        gatherer_examples = load_training_examples(str(RESEARCH_DATAGATHERER_DATA_PATH))
        worker_training_examples["DataGatherer"] = gatherer_examples
        print(f"Loaded {len(gatherer_examples)} DataGatherer training examples")

    if RESEARCH_SUMMARIZER_DATA_PATH.exists():
        summarizer_examples = load_training_examples(str(RESEARCH_SUMMARIZER_DATA_PATH))
        worker_training_examples["Summarizer"] = summarizer_examples
        print(f"Loaded {len(summarizer_examples)} Summarizer training examples")

    # Create optimizer with both manager and worker optimization enabled
    optimizer = MultiAgentOptimizer(
        runner,
        optimize_manager=True,
        optimize_workers=True,  # Enable recursive worker optimization
    )

    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    print("\nStarting recursive multi-agent optimization...")
    print("-" * 40)

    result = optimizer.compile(
        training_examples=manager_examples,
        metric=research_quality_metric,
        worker_training_examples=worker_training_examples,  # Per-worker training data
        # worker_metrics={...},  # Optional: per-worker custom metrics
        optimizer="mipro",
        max_bootstrapped_demos=2,
        dspy_lm=dspy_lm,
        mipro_auto='light',
    )

    result.save("research_team_optimized.json")

    print()
    print("=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)
    print(f"Config saved to: research_team_optimized.json")
    print(f"Optimization runs: {len(result.optimization.runs)}")

    # Show what was optimized
    last_run = result.optimization.runs[-1]
    print(f"\nManager optimization:")
    print(f"  - role_definition changed: {last_run.role_definition_changed}")
    print(f"  - examples: {last_run.examples_before} -> {last_run.examples_after}")

    workers_optimized = last_run.optimizer_config.get("workers_optimized", [])
    if workers_optimized:
        print(f"\nWorkers optimized: {workers_optimized}")
        for worker_name in workers_optimized:
            worker_config = result.config.get("workers", {}).get(worker_name, {})
            worker_prompts = worker_config.get("prompts", {})
            worker_examples = worker_prompts.get("examples", [])
            print(f"  - {worker_name}: {len(worker_examples)} examples")

    return runner


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
    # print("Running: example_optimize_agent")
    # asyncio.run(example_optimize_agent())
    
    # Example 2: Optimize a simple LLM (no agent)
    print("Running: example_optimize_simple_llm")
    asyncio.run(example_optimize_simple_llm())
    
    # Example 3: Optimize a multi-agent system
    # print("Running: example_optimize_multi_agent")
    # asyncio.run(example_optimize_multi_agent())