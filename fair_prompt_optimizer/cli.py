# fair_prompt_optimizer/cli.py
"""
Command-line interface for FAIR Prompt Optimizer.

Usage:
    fair-optimize init [--type TYPE] [--output PATH]
    fair-optimize optimize -c CONFIG -t TRAINING [OPTIONS]
    fair-optimize test -c CONFIG [-i INPUT]
    fair-optimize info -c CONFIG
    fair-optimize compare CONFIG1 CONFIG2
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="fair-optimize",
        description="FAIR Prompt Optimizer - DSPy-powered optimization for FAIR-LLM agents",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- init command ---
    init_parser = subparsers.add_parser("init", help="Initialize a new config file")
    init_parser.add_argument(
        "--type", "-t",
        choices=["simple_llm", "agent", "multi_agent"],
        default="agent",
        help="Type of config to create (default: agent)"
    )
    init_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: {type}_config.json)"
    )
    
    # --- optimize command ---
    opt_parser = subparsers.add_parser("optimize", help="Optimize an agent's prompts")
    opt_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to agent config JSON"
    )
    opt_parser.add_argument(
        "--training", "-t",
        required=True,
        help="Path to training examples JSON"
    )
    opt_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for optimized config (default: {config}_optimized.json)"
    )
    opt_parser.add_argument(
        "--optimizer",
        choices=["bootstrap", "mipro"],
        default="bootstrap",
        help="DSPy optimizer to use (default: bootstrap)"
    )
    opt_parser.add_argument(
        "--metric",
        choices=["exact_match", "contains_answer", "numeric_accuracy", "fuzzy_match"],
        default="contains_answer",
        help="Evaluation metric (default: contains_answer)"
    )
    opt_parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Max bootstrapped demos (default: 4)"
    )
    opt_parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (overrides config)"
    )
    opt_parser.add_argument(
        "--adapter",
        choices=["HuggingFaceAdapter", "OpenAIAdapter", "OllamaAdapter"],
        default=None,
        help="LLM adapter to use (overrides config)"
    )
    opt_parser.add_argument(
        "--mipro-lm",
        default=None,
        help="DSPy LM for MIPROv2 (e.g., 'ollama_chat/llama3:8b')"
    )
    opt_parser.add_argument(
        "--mipro-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 intensity (default: light)"
    )
    
    # --- test command ---
    test_parser = subparsers.add_parser("test", help="Test an agent interactively")
    test_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to agent config JSON"
    )
    test_parser.add_argument(
        "--input", "-i",
        default=None,
        help="Single input to test (if not provided, enters interactive mode)"
    )
    test_parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (overrides config)"
    )
    test_parser.add_argument(
        "--adapter",
        default=None,
        help="LLM adapter to use (overrides config)"
    )
    
    # --- info command ---
    info_parser = subparsers.add_parser("info", help="Show config information")
    info_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config JSON"
    )
    info_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full config details"
    )
    
    # --- compare command ---
    compare_parser = subparsers.add_parser("compare", help="Compare two configs")
    compare_parser.add_argument("config1", help="First config file")
    compare_parser.add_argument("config2", help="Second config file")
    
    # --- examples command ---
    examples_parser = subparsers.add_parser("examples", help="Create example training data")
    examples_parser.add_argument(
        "--output", "-o",
        default="examples.json",
        help="Output file path"
    )
    examples_parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="Number of example templates to create"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handlers
    try:
        if args.command == "init":
            cmd_init(args)
        elif args.command == "optimize":
            cmd_optimize(args)
        elif args.command == "test":
            cmd_test(args)
        elif args.command == "info":
            cmd_info(args)
        elif args.command == "compare":
            cmd_compare(args)
        elif args.command == "examples":
            cmd_examples(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Command: init
# =============================================================================

def cmd_init(args):
    """Create a new config file from template."""
    
    output_path = args.output or f"{args.type}_config.json"
    
    if args.type == "simple_llm":
        config = create_simple_llm_template()
    elif args.type == "agent":
        config = create_agent_template()
    elif args.type == "multi_agent":
        config = create_multi_agent_template()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created {args.type} config: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {output_path} to configure your agent")
    print(f"  2. Create training examples: fair-optimize examples -o examples.json")
    print(f"  3. Optimize: fair-optimize optimize -c {output_path} -t examples.json")


def create_simple_llm_template() -> dict:
    """Create a simple_llm config template."""
    return {
        "version": "1.0",
        "type": "simple_llm",
        "prompts": {
            "role_definition": "# TODO: Define your system prompt\n# Example: You are a helpful classifier. Always respond with: CATEGORY: [label]",
            "tool_instructions": [],
            "worker_instructions": [],
            "format_instructions": [
                "# TODO: Add format instructions if needed"
            ],
            "examples": []
        },
        "model": {
            "adapter": "HuggingFaceAdapter",
            "model_name": "# TODO: Set your model name (e.g., dolphin3-qwen25-3b)",
            "adapter_kwargs": {}
        },
        "agent": {
            "planner_type": "None",
            "tools": [],
            "max_steps": 1,
            "stateless": True
        }
    }


def create_agent_template() -> dict:
    """Create an agent config template."""
    return {
        "version": "1.0",
        "type": "agent",
        "prompts": {
            "role_definition": "# TODO: Define your agent's role and goal\n# Example: You are a helpful research assistant with access to tools.",
            "tool_instructions": [
                {
                    "name": "# TODO: tool_name",
                    "description": "# TODO: What this tool does"
                }
            ],
            "worker_instructions": [],
            "format_instructions": [
                "Always show your reasoning in a 'Thought' section.",
                "Provide actions as JSON with 'tool_name' and 'tool_input'.",
                "When you have the final answer, use the 'final_answer' action."
            ],
            "examples": []
        },
        "model": {
            "adapter": "HuggingFaceAdapter",
            "model_name": "# TODO: Set your model name",
            "adapter_kwargs": {}
        },
        "agent": {
            "planner_type": "SimpleReActPlanner",
            "tools": [
                "# TODO: List tool class names (e.g., SafeCalculatorTool)"
            ],
            "max_steps": 10,
            "stateless": False
        }
    }


def create_multi_agent_template() -> dict:
    """Create a multi_agent config template."""
    return {
        "version": "1.0",
        "type": "multi_agent",
        "manager": {
            "prompts": {
                "role_definition": "# TODO: Define manager's role\n# Example: You are a project manager. Analyze requests and delegate to specialist workers.",
                "tool_instructions": [],
                "worker_instructions": [
                    {
                        "name": "# TODO: Worker1",
                        "role_description": "# TODO: What this worker does"
                    },
                    {
                        "name": "# TODO: Worker2", 
                        "role_description": "# TODO: What this worker does"
                    }
                ],
                "format_instructions": [
                    "Use 'delegate' to assign tasks: {\"tool_name\": \"delegate\", \"tool_input\": {\"worker_name\": \"...\", \"task\": \"...\"}}",
                    "Use 'final_answer' when complete: {\"tool_name\": \"final_answer\", \"tool_input\": \"...\"}"
                ],
                "examples": []
            },
            "model": {
                "adapter": "HuggingFaceAdapter",
                "model_name": "# TODO: Set model"
            },
            "agent": {
                "planner_type": "ManagerPlanner",
                "tools": [],
                "max_steps": 15
            }
        },
        "workers": {
            "Worker1": {
                "prompts": {
                    "role_definition": "# TODO: Define worker's specialty",
                    "tool_instructions": [],
                    "format_instructions": [],
                    "examples": []
                },
                "model": {
                    "adapter": "HuggingFaceAdapter",
                    "model_name": "# TODO: Set model"
                },
                "agent": {
                    "planner_type": "SimpleReActPlanner",
                    "tools": [],
                    "max_steps": 5,
                    "stateless": True
                }
            }
        },
        "max_delegation_steps": 15
    }


# =============================================================================
# Command: optimize
# =============================================================================

def cmd_optimize(args):
    """Run optimization on an agent."""
    import nest_asyncio
    nest_asyncio.apply()
    
    from .config import (
        load_optimized_config,
        load_training_examples,
        save_optimized_config,
        compute_file_hash,
    )
    from .optimizers import AgentOptimizer, MultiAgentOptimizer, SimpleLLMOptimizer
    from . import metrics as metrics_module
    
    # Load config
    print(f"Loading config: {args.config}")
    config = load_optimized_config(args.config)
    config_type = config.type
    
    # Load training examples
    print(f"Loading training examples: {args.training}")
    examples = load_training_examples(args.training)
    print(f"  Found {len(examples)} examples")
    
    # Create LLM
    llm = create_llm(
        config.config.get("model", {}),
        model_override=args.model,
        adapter_override=args.adapter,
    )
    
    # Get metric function
    metric = getattr(metrics_module, args.metric)
    print(f"Using metric: {args.metric}")
    
    # Create optimizer based on type
    print(f"Config type: {config_type}")
    
    if config_type == "simple_llm":
        system_prompt = config.prompts.get("role_definition", "You are a helpful assistant.")
        optimizer = SimpleLLMOptimizer(llm, system_prompt)
        # Transfer existing config
        optimizer.config = config
        
    elif config_type == "multi_agent":
        from fairlib.utils.config_manager import load_multi_agent
        runner = load_multi_agent(args.config, llm)
        optimizer = MultiAgentOptimizer(runner, config=config)
        
    else:  # agent
        from fairlib.utils.config_manager import load_agent
        agent = load_agent(args.config, llm)
        optimizer = AgentOptimizer(agent, config=config)
    
    # Setup MIPROv2 if needed
    dspy_lm = None
    if args.optimizer == "mipro":
        if args.mipro_lm:
            import dspy
            dspy_lm = dspy.LM(args.mipro_lm)
        else:
            print("Warning: MIPROv2 works best with --mipro-lm specified")
    
    # Run optimization
    print()
    print(f"Running {args.optimizer} optimization...")
    print(f"  Max demos: {args.max_demos}")
    if args.optimizer == "mipro":
        print(f"  MIPROv2 auto: {args.mipro_auto}")
    print()
    
    result = optimizer.compile(
        training_examples=examples,
        metric=metric,
        optimizer=args.optimizer,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
        training_data_path=args.training,
        dspy_lm=dspy_lm,
        mipro_auto=args.mipro_auto,
    )
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        base = Path(args.config).stem
        output_path = f"{base}_optimized.json"
    
    # Save
    save_optimized_config(result, output_path)
    
    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {output_path}")
    print(f"Examples generated: {len(result.examples)}")
    print(f"Optimization runs: {len(result.optimization.runs)}")
    print()
    print("Next steps:")
    print(f"  Test: fair-optimize test -c {output_path}")
    print(f"  Info: fair-optimize info -c {output_path}")


# =============================================================================
# Command: test
# =============================================================================

def cmd_test(args):
    """Test an agent interactively."""
    import nest_asyncio
    nest_asyncio.apply()
    
    from .config import load_optimized_config
    
    # Load config
    config = load_optimized_config(args.config)
    config_type = config.type
    
    # Create LLM
    llm = create_llm(
        config.config.get("model", {}),
        model_override=args.model,
        adapter_override=args.adapter,
    )
    
    print(f"Loaded {config_type} config: {args.config}")
    
    # Create agent/optimizer for testing
    if config_type == "simple_llm":
        from .optimizers import SimpleLLMOptimizer
        system_prompt = config.prompts.get("role_definition", "")
        tester = SimpleLLMOptimizer(llm, system_prompt)
        
    elif config_type == "multi_agent":
        from fairlib.utils.config_manager import load_multi_agent
        from .optimizers import MultiAgentOptimizer
        runner = load_multi_agent(args.config, llm)
        tester = MultiAgentOptimizer(runner, config=config)
        
    else:  # agent
        from fairlib.utils.config_manager import load_agent
        from .optimizers import AgentOptimizer
        agent = load_agent(args.config, llm)
        tester = AgentOptimizer(agent, config=config)
    
    # Single input mode
    if args.input:
        print(f"\nInput: {args.input}")
        print("-" * 40)
        result = tester.test(args.input)
        print(f"Output: {result}")
        return
    
    # Interactive mode
    print()
    print("Interactive test mode. Type 'quit' or 'exit' to stop.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            result = tester.test(user_input)
            print(f"\nAgent: {result}")
            
        except EOFError:
            print("\nGoodbye!")
            break


# =============================================================================
# Command: info
# =============================================================================

def cmd_info(args):
    """Show information about a config."""
    from .config import load_optimized_config
    
    config = load_optimized_config(args.config)
    
    print()
    print("=" * 60)
    print(f"CONFIG: {args.config}")
    print("=" * 60)
    
    print(f"\nType: {config.type}")
    print(f"Version: {config.config.get('version', 'unknown')}")
    
    # Model info
    model = config.config.get("model", {})
    print(f"\nModel:")
    print(f"  Adapter: {model.get('adapter', 'unknown')}")
    print(f"  Name: {model.get('model_name', 'unknown')}")
    
    # Prompts
    prompts = config.prompts
    print(f"\nPrompts:")
    print(f"  Role definition: {len(prompts.get('role_definition', '') or '')} chars")
    print(f"  Tool instructions: {len(prompts.get('tool_instructions', []))}")
    print(f"  Worker instructions: {len(prompts.get('worker_instructions', []))}")
    print(f"  Format instructions: {len(prompts.get('format_instructions', []))}")
    print(f"  Examples: {len(prompts.get('examples', []))}")
    
    # Agent spec
    agent = config.config.get("agent", {})
    if agent:
        print(f"\nAgent:")
        print(f"  Planner: {agent.get('planner_type', 'unknown')}")
        print(f"  Tools: {agent.get('tools', [])}")
        print(f"  Max steps: {agent.get('max_steps', 'unknown')}")
    
    # Optimization provenance
    opt = config.optimization
    print(f"\nOptimization:")
    print(f"  Optimized: {opt.optimized}")
    if opt.optimized:
        print(f"  Optimizer: {opt.optimizer}")
        print(f"  Metric: {opt.metric}")
        print(f"  Last optimized: {opt.last_optimized_at}")
        print(f"  Total runs: {len(opt.runs)}")
        
        # Show training data hash from most recent run (FIXED: was opt.training_data.path)
        last_run = opt.runs[-1]
        if last_run.training_data_hash:
            print(f"  Training data hash: {last_run.training_data_hash}")
        
        # Show run history summary
        if len(opt.runs) > 1:
            print(f"\n  Run history:")
            for i, run in enumerate(opt.runs, 1):
                print(f"    {i}. {run.optimizer} ({run.timestamp[:10]}) - {run.examples_before}→{run.examples_after} examples")
    
    # Verbose output
    if args.verbose:
        print("\n" + "=" * 60)
        print("FULL CONFIG")
        print("=" * 60)
        print(json.dumps(config.to_dict(), indent=2))
    
    # Multi-agent specific
    if config.type == "multi_agent":
        workers = config.config.get("workers", {})
        print(f"\nWorkers: {list(workers.keys())}")


# =============================================================================
# Command: compare
# =============================================================================

def cmd_compare(args):
    """Compare two configs."""
    from .config import load_optimized_config
    
    config1 = load_optimized_config(args.config1)
    config2 = load_optimized_config(args.config2)
    
    print()
    print("=" * 70)
    print("CONFIG COMPARISON")
    print("=" * 70)
    print(f"\nConfig 1: {args.config1}")
    print(f"Config 2: {args.config2}")
    
    # Compare prompts
    prompts1 = config1.prompts
    prompts2 = config2.prompts
    
    print("\n" + "-" * 70)
    print("PROMPTS")
    print("-" * 70)
    
    # Role definition
    role1 = prompts1.get("role_definition", "")
    role2 = prompts2.get("role_definition", "")
    if role1 != role2:
        print(f"\n[CHANGED] Role definition:")
        print(f"  Config 1: {(role1 or '')[:100]}{'...' if len(role1 or '') > 100 else ''}")
        print(f"  Config 2: {(role2 or '')[:100]}{'...' if len(role2 or '') > 100 else ''}")
    else:
        print(f"\n[SAME] Role definition ({len(role1 or '')} chars)")
    
    # Examples
    ex1 = prompts1.get("examples", [])
    ex2 = prompts2.get("examples", [])
    print(f"\n[{'CHANGED' if ex1 != ex2 else 'SAME'}] Examples: {len(ex1)} → {len(ex2)}")
    
    # Format instructions
    fi1 = prompts1.get("format_instructions", [])
    fi2 = prompts2.get("format_instructions", [])
    print(f"[{'CHANGED' if fi1 != fi2 else 'SAME'}] Format instructions: {len(fi1)} → {len(fi2)}")
    
    # Optimization comparison
    print("\n" + "-" * 70)
    print("OPTIMIZATION")
    print("-" * 70)
    
    opt1 = config1.optimization
    opt2 = config2.optimization
    
    print(f"\nConfig 1: {'Optimized' if opt1.optimized else 'Not optimized'}")
    if opt1.optimized:
        print(f"  Optimizer: {opt1.optimizer}")
        print(f"  Runs: {len(opt1.runs)}")
    
    print(f"\nConfig 2: {'Optimized' if opt2.optimized else 'Not optimized'}")
    if opt2.optimized:
        print(f"  Optimizer: {opt2.optimizer}")
        print(f"  Runs: {len(opt2.runs)}")


# =============================================================================
# Command: examples
# =============================================================================

def cmd_examples(args):
    """Create example training data file."""
    
    examples = []
    for i in range(args.count):
        examples.append({
            "inputs": {
                "user_input": f"# TODO: Example {i+1} input"
            },
            "expected_output": f"# TODO: Example {i+1} expected output"
        })
    
    # Add helpful comments
    template = {
        "_comment": "Training examples for fair_prompt_optimizer. Each example has 'inputs' (dict with user_input) and 'expected_output' (string).",
        "_tips": [
            "Include 10-50 diverse examples for best results",
            "Mix easy and hard cases",
            "Include edge cases and error handling examples",
            "Make expected_output match what you want the agent to produce"
        ],
        "examples": examples
    }
    
    # Save as just the examples array (without metadata for actual use)
    with open(args.output, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Created training examples template: {args.output}")
    print(f"  {args.count} example templates created")
    print()
    print("Edit the file to add your actual training examples.")
    print("Each example needs:")
    print('  - inputs.user_input: The input prompt')
    print('  - expected_output: What the agent should produce')


# =============================================================================
# Helpers
# =============================================================================

def create_llm(model_config: dict, model_override: str = None, adapter_override: str = None):
    """Create an LLM from config with optional overrides."""
    
    adapter_name = adapter_override or model_config.get("adapter", "HuggingFaceAdapter")
    model_name = model_override or model_config.get("model_name", "dolphin3-qwen25-3b")
    adapter_kwargs = model_config.get("adapter_kwargs", {})
    
    # Import adapter
    if adapter_name == "HuggingFaceAdapter":
        from fairlib import HuggingFaceAdapter
        return HuggingFaceAdapter(model_name, **adapter_kwargs)
    elif adapter_name == "OpenAIAdapter":
        from fairlib import OpenAIAdapter
        return OpenAIAdapter(model_name, **adapter_kwargs)
    elif adapter_name == "OllamaAdapter":
        from fairlib import OllamaAdapter
        return OllamaAdapter(model_name, **adapter_kwargs)
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()