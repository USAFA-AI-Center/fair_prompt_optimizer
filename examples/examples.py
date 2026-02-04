#!/usr/bin/env python3
"""
FAIR Prompt Optimizer - Examples

This file has been split into separate example files for clarity:

1. 01_simple_llm_example.py  - SimpleLLMOptimizer (classification, format compliance)
2. 02_agent_example.py       - AgentOptimizer (tool-using agents with ReAct)
3. 03_multi_agent_example.py - MultiAgentOptimizer (manager-worker teams)

Run individual examples:
    cd /home/ai-user/fair_prompt_optimizer
    source .venv/bin/activate
    python examples/01_simple_llm_example.py
    python examples/02_agent_example.py
    python examples/03_multi_agent_example.py

Or run this file to execute all examples sequentially.
"""

import asyncio
import sys
from pathlib import Path


async def run_all_examples():
    """Run all example files sequentially."""
    examples_dir = Path(__file__).parent

    examples = [
        ("Simple LLM Optimization", "01_simple_llm_example.py"),
        ("Agent Optimization", "02_agent_example.py"),
        ("Multi-Agent Optimization", "03_multi_agent_example.py"),
    ]

    print("=" * 60)
    print("FAIR PROMPT OPTIMIZER - ALL EXAMPLES")
    print("=" * 60)
    print()
    print("Examples to run:")
    for name, filename in examples:
        print(f"  - {name}: {filename}")
    print()

    for name, filename in examples:
        print()
        print("=" * 60)
        print(f"Running: {name}")
        print("=" * 60)
        print()

        # Import and run each example
        example_path = examples_dir / filename

        if example_path.exists():
            # Dynamic import
            import importlib.util
            spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), example_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[filename.replace(".py", "")] = module
            spec.loader.exec_module(module)

            # Run the main function
            if hasattr(module, 'main'):
                await module.main()
            else:
                print(f"Warning: {filename} has no main() function")
        else:
            print(f"Warning: {filename} not found")

        print()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Check if user wants to run a specific example
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        print(f"To run a specific example, use:")
        print(f"  python examples/01_simple_llm_example.py")
        print(f"  python examples/02_agent_example.py")
        print(f"  python examples/03_multi_agent_example.py")
    else:
        asyncio.run(run_all_examples())
