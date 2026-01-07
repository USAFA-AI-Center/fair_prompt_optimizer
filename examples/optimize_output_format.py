#!/usr/bin/env python3
"""
Optimizing a FAIR-LLM agent with format-constrained output.

This demonstrates:
1. Custom metric that checks BOTH format and numeric accuracy
2. Training examples with good/bad format (optimizer learns to select good ones)
3. Loading and testing the optimized agent
"""

import re
import gc
import asyncio
from typing import Optional

import torch

from fair_prompt_optimizer.metrics import _get_expected, _get_predicted
from fair_prompt_optimizer import (
    FAIRPromptOptimizer,
    load_optimized_agent,
    load_training_examples,
)

def extract_number(text: str) -> Optional[float]:
    """Extract the last number from text."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def format_and_numeric(example, prediction, trace=None) -> bool:
    """
    Custom metric that requires BOTH:
    1. Output starts with "ANSWER:"
    2. Numeric value is correct (within 1% tolerance)
    
    This filters out:
    - Wrong format: "The answer is 42", "42", "Result: 42"
    - Wrong value: "ANSWER: 100" when expected "ANSWER: 42"
    """
    predicted = _get_predicted(prediction)
    expected = _get_expected(example)
    
    if predicted is None or expected is None:
        return False
    
    # Check 1: Format compliance
    if not predicted.strip().upper().startswith("ANSWER:"):
        return False
    
    # Check 2: Numeric accuracy
    expected_num = extract_number(expected)
    predicted_num = extract_number(predicted)
    
    if expected_num is None or predicted_num is None:
        return False
    
    if expected_num == 0:
        return abs(predicted_num) < 0.01
    
    rel_diff = abs(expected_num - predicted_num) / abs(expected_num)
    return rel_diff <= 0.01

def main():
    print("\n" + "=" * 60)
    print("FAIR Prompt Optimizer - Format Constraint Demo")
    print("=" * 60)
    
    # --- Step 1: Load the FAIR-LLM agent from config ---
    print("\n[1/3] Loading FAIR-LLM agent from config...")
    agent = load_optimized_agent("output_format_example_files/agent_config.json")
    print("  ✓ Agent loaded")
    
    # --- Step 2: Load training examples ---
    print("\n[2/3] Loading training examples...")
    examples = load_training_examples("output_format_example_files/examples.json")
    print(f"  ✓ Loaded {len(examples)} examples")
    
    # --- Step 3: Run optimization ---
    print("\n[3/3] Running BootstrapFewShot optimization...")
    print("  Using custom metric: format_and_numeric")
    print("  This will select examples that have correct format AND value\n")
    
    optimizer = FAIRPromptOptimizer(agent)
    
    try:
        optimized_config = optimizer.compile(
            training_examples=examples,
            metric=format_and_numeric,
            optimizer="bootstrap",
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            output_path="./formatted_output_agent/agent_config_optimized.json"
        )
        
        # --- Report results ---
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        
        return optimized_config
        
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        raise

def test_optimized_agent():
    """
    Load and test the optimized agent interactively.
    """
    print("\n" + "=" * 60)
    print("Testing Optimized Agent")
    print("=" * 60)
    print("Type 'exit' to quit.\n")
    
    # Load agent from saved config
    agent = load_optimized_agent("./formatted_output_agent/agent_config_optimized.json")
    
    async def run_loop():
        while True:
            try:
                user_input = input("👤 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("🤖 Agent: Goodbye!")
                    break
                
                response = await agent.arun(user_input)
                print(f"🤖 Agent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n🤖 Agent: Exiting...")
                break
    
    asyncio.run(run_loop())

if __name__ == "__main__":
    config = main()
    
    # Optional: test the optimized agent
    print("\n" + "-" * 60)
    response = input("Test the optimized agent? [y/N]: ")
    if response.lower() == 'y':
        test_optimized_agent()