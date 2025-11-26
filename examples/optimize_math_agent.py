"""
Example: Optimizing a Math Agent
================================

This example demonstrates how to optimize prompts for a FAIR-LLM math agent
using the fair_prompt_optimizer.

Prerequisites:
    - FAIR-LLM installed with a math agent configured
    - OpenAI API key set in environment (OPENAI_API_KEY)

Usage:
    python examples/optimize_math_agent.py
"""

import json
from pathlib import Path
import os
import litellm

from fair_prompt_optimizer.optimizers import FAIRPromptOptimizer
from fair_prompt_optimizer.optimizers import TrainingExample
from fair_prompt_optimizer.metrics import exact_match
from fair_prompt_optimizer.metrics import numeric_accuracy

def _check_ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

def get_model_config():
    litellm.drop_params = True
    
    if not _check_ollama_available():
        raise EnvironmentError(
            "Ollama not running!\n\n"
            "Start it with:\n"
            "  ollama serve\n\n"
            "And make sure you have Llama pulled:\n"
            "  ollama pull llama3.1:8b"
        )
    
    print("🦙 Using Ollama with Llama 3.1 8B")
    return {
        "lm_model": "ollama/llama3.1:8b",
        "lm_kwargs": {
            "api_base": "http://localhost:11434",
            "max_tokens": 1024,
        }
    }

def main():
    print("=" * 60)
    print("   Math Agent Prompt Optimization Example")
    print("=" * 60)
    print()
    
    base_config = {
        "version": "1.0",
        "role_definition": (
            "You are an expert mathematical calculator. Your job is to perform "
            "mathematical calculations accurately. You reason step-by-step to "
            "determine the best course of action."
        ),
        "tool_instructions": [
            {
                "name": "safe_calculator",
                "description": "Performs safe mathematical calculations. Input should be a valid mathematical expression."
            },
            {
                "name": "final_answer", 
                "description": "Use this tool to provide the final answer to the user."
            }
        ],
        "worker_instructions": [],
        "format_instructions": [
            "Your response must contain a 'Thought' explaining your reasoning.",
            "Your response must contain an 'Action' with a tool call or final answer.",
            "Always show your work before giving the final answer."
        ],
        "examples": [],
        "metadata": {
            "optimized": False
        }
    }
    
    # Save the base config
    base_path = Path("prompts")
    base_path.mkdir(exist_ok=True)
    
    with open(base_path / "math_agent_base.json", "w") as f:
        json.dump(base_config, f, indent=2)
    
    print(f"Created base configuration: {base_path / 'math_agent_base.json'}")
    
    training_examples = [
        TrainingExample(
            inputs={"user_query": "What is 15 + 27?"},
            expected_output="42"
        ),
        TrainingExample(
            inputs={"user_query": "Calculate 100 divided by 4"},
            expected_output="25"
        ),
        TrainingExample(
            inputs={"user_query": "What is 8 times 7?"},
            expected_output="56"
        ),
        TrainingExample(
            inputs={"user_query": "What is 144 minus 89?"},
            expected_output="55"
        ),
        TrainingExample(
            inputs={"user_query": "What is 25% of 200?"},
            expected_output="50"
        ),
        TrainingExample(
            inputs={"user_query": "Calculate the square root of 81"},
            expected_output="9"
        ),
        TrainingExample(
            inputs={"user_query": "What is 2 to the power of 8?"},
            expected_output="256"
        ),
        TrainingExample(
            inputs={"user_query": "What is (10 + 5) * 3?"},
            expected_output="45"
        ),
        TrainingExample(
            inputs={"user_query": "If I have 45 apples and give away 12, then buy 8 more, how many do I have?"},
            expected_output="41"
        ),
        TrainingExample(
            inputs={"user_query": "What is 1000 divided by 8, rounded to the nearest whole number?"},
            expected_output="125"
        ),
        TrainingExample(
            inputs={"user_query": "A rectangle has length 12 and width 5. What is its area?"},
            expected_output="60"
        ),
        TrainingExample(
            inputs={"user_query": "If a car travels at 60 mph for 2.5 hours, how many miles does it travel?"},
            expected_output="150"
        ),
    ]
    
    print(f"Created {len(training_examples)} training examples")
    print()
    print("Initializing optimizer...")

    model_config = get_model_config()
    optimizer = FAIRPromptOptimizer(
        lm_model=model_config["lm_model"],
        **model_config["lm_kwargs"]
    )
    
    print("Optimizer initialized")
    print()
    print("Running BootstrapFewShot optimization...")
    print("   This generates high-quality few-shot examples from your training data.")
    print()
    
    bootstrap_result = optimizer.optimize_bootstrap(
        fair_config_path=base_path / "math_agent_base.json",
        training_examples=training_examples,
        metric=numeric_accuracy,  # Use numeric comparison for math
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        output_path=base_path / "math_agent_bootstrap.json"
    )
    
    print(f"✅ Bootstrap optimization complete!")
    print(f"   Duration: {bootstrap_result.duration_seconds:.2f}s")
    print(f"   Examples generated: {len(bootstrap_result.config.examples)}")
    print(f"   Output: {bootstrap_result.output_path}")
    
    print()
    print("🚀 Running MIPROv2 optimization...")
    print("   This jointly optimizes instructions AND few-shot examples.")
    print()
    
    try:
        mipro_result = optimizer.optimize_mipro(
            fair_config_path=base_path / "math_agent_base.json",
            training_examples=training_examples,
            metric=numeric_accuracy,
            auto="light",  # Use "medium" or "heavy" for better results (more compute)
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            output_path=base_path / "math_agent_mipro.json"
        )
        
        print(f"MIPROv2 optimization complete!")
        print(f"   Duration: {mipro_result.duration_seconds:.2f}s")
        print(f"   Examples generated: {len(mipro_result.config.examples)}")
        print(f"   Output: {mipro_result.output_path}")
        
    except ImportError as e:
        print(f"MIPROv2 not available: {e}")
        print("   Upgrade DSPy to use MIPROv2: pip install --upgrade dspy-ai")
    
    print()
    print("=" * 60)
    print("   Optimization Results Summary")
    print("=" * 60)
    print()
    
    # Load and display the optimized configs
    with open(base_path / "math_agent_bootstrap.json") as f:
        bootstrap_config = json.load(f)
    
    print("📊 Bootstrap Optimized Config:")
    print(f"   Role definition: {bootstrap_config['role_definition'][:100]}...")
    print(f"   Number of examples: {len(bootstrap_config['examples'])}")
    if bootstrap_config['examples']:
        print(f"   First example preview:")
        print(f"      {bootstrap_config['examples'][0][:200]}...")

if __name__ == "__main__":
    main()
