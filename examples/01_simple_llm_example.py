#!/usr/bin/env python3
"""
Example 1: Optimize a Simple LLM (No Agent)

Use case: Classification, format compliance, simple generation tasks.

This example shows how to optimize a base LLM with just a system prompt,
without any agent pipeline or tools.

Usage:
    cd /home/ai-user/fair_prompt_optimizer
    source .venv/bin/activate
    python examples/01_simple_llm_example.py
"""

# Suppress warnings from unneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
from pathlib import Path
import dspy

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Training data path
SENTIMENT_DATA_PATH = Path(__file__).parent / "training_data" / "sentiment_examples.json"


async def main():
    """
    Optimize a simple LLM for sentiment classification.
    """
    from fairlib import HuggingFaceAdapter

    from fair_prompt_optimizer import (
        SimpleLLMOptimizer,
        load_training_examples,
        format_compliance,
    )

    print("=" * 60)
    print("EXAMPLE: OPTIMIZE SIMPLE LLM (No Agent)")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # Step 1: Create the LLM and initial system prompt
    # -------------------------------------------------------------------------
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    system_prompt = """You are a sentiment classifier.

# RESPONSE FORMAT (MANDATORY)
Respond with ONLY one of:
SENTIMENT: positive
SENTIMENT: negative
SENTIMENT: neutral

No explanations, just the label."""

    print("Initial system prompt:")
    print("-" * 40)
    print(system_prompt)
    print()

    # -------------------------------------------------------------------------
    # Step 2: Load training examples
    # -------------------------------------------------------------------------
    if not SENTIMENT_DATA_PATH.exists():
        print(f"ERROR: Training data not found at {SENTIMENT_DATA_PATH}")
        return

    training_examples = load_training_examples(str(SENTIMENT_DATA_PATH))
    print(f"Loaded {len(training_examples)} training examples")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Create optimizer and run optimization
    # -------------------------------------------------------------------------
    optimizer = SimpleLLMOptimizer(llm=llm, system_prompt=system_prompt)

    # DSPy LM for generating MIPRO candidate prompts
    dspy_lm = dspy.LM(
        'ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )

    print("Starting optimization with MIPRO (light)...")
    print("-" * 40)

    result = optimizer.compile(
        training_examples=training_examples,
        metric=format_compliance("SENTIMENT:"),
        optimizer="mipro",
        mipro_auto="light",
        dspy_lm=dspy_lm,
        max_bootstrapped_demos=3,
    )

    # -------------------------------------------------------------------------
    # Step 4: Save and show results
    # -------------------------------------------------------------------------
    output_path = Path(__file__).parent / "classifier_optimized.json"
    result.save(str(output_path))

    print()
    print("=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)
    print(f"Optimized examples: {len(result.prompts.get('examples', []))}")
    print(f"Saved to: {output_path}")
    print()

    # Show optimization details
    if result.optimization and result.optimization.runs:
        last_run = result.optimization.runs[-1]
        print(f"Optimizer: {last_run.optimizer}")
        print(f"Role definition changed: {last_run.role_definition_changed}")
        print(f"Format instructions changed: {last_run.format_instructions_changed}")
        print(f"Examples: {last_run.examples_before} -> {last_run.examples_after}")
        print()

    # -------------------------------------------------------------------------
    # Step 5: Test the optimized classifier
    # -------------------------------------------------------------------------
    print("Testing optimized classifier:")
    print("-" * 40)

    test_inputs = [
        "This product is amazing! Best purchase ever!",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
    ]

    for text in test_inputs:
        response = optimizer.test(text)
        print(f"Input: {text}")
        print(f"Output: {response}")
        print()

    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
