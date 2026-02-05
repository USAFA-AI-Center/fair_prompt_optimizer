# fair_prompt_optimizer/cli.py
"""
Command-line interface for FAIR Prompt Optimizer.

Usage:
    fair-optimize init [--type TYPE] [--output PATH]
    fair-optimize optimize -c CONFIG -t TRAINING [OPTIONS]
    fair-optimize test -c CONFIG [-i INPUT]
    fair-optimize info -c CONFIG
    fair-optimize compare CONFIG1 CONFIG2
    fair-optimize validate -c CONFIG [-t TRAINING]
"""

import argparse
import json
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# Terminal Output Helpers
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.MAGENTA = cls.CYAN = cls.BOLD = cls.DIM = cls.RESET = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_error(msg: str, suggestion: str = None):
    """Print an error message with optional suggestion."""
    print(f"{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {msg}", file=sys.stderr)
    if suggestion:
        print(f"{Colors.DIM}  Suggestion: {suggestion}{Colors.RESET}", file=sys.stderr)


def print_warning(msg: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}Warning:{Colors.RESET} {msg}", file=sys.stderr)


def print_success(msg: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_info(msg: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_step(step: int, total: int, msg: str):
    """Print a step progress message."""
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.RESET} {msg}")


class Spinner:
    """Simple terminal spinner for long-running operations."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Working..."):
        self.message = message
        self.running = False
        self.thread = None
        self.frame_idx = 0

    def _spin(self):
        while self.running:
            frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
            sys.stdout.write(f"\r{Colors.CYAN}{frame}{Colors.RESET} {self.message}")
            sys.stdout.flush()
            self.frame_idx += 1
            time.sleep(0.1)

    def start(self):
        if sys.stdout.isatty():
            self.running = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        else:
            print(f"  {self.message}")

    def stop(self, success: bool = True, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        if sys.stdout.isatty():
            # Clear the spinner line
            sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
            sys.stdout.flush()
        if final_message:
            if success:
                print_success(final_message)
            else:
                print_error(final_message)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# =============================================================================
# Validation Functions
# =============================================================================


def validate_config(config_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a config file.

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Check file exists
    if not Path(config_path).exists():
        return False, [f"Config file not found: {config_path}"], []

    # Try to load JSON
    try:
        with open(config_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in config: {e}"], []

    # Check required fields
    if "type" not in data:
        warnings.append("Missing 'type' field - defaulting to 'agent'")

    config_type = data.get("type", "agent")

    # Check prompts
    prompts = data.get("prompts", {})
    if not prompts:
        if config_type != "multi_agent":
            errors.append("Missing 'prompts' section")
    else:
        role_def = prompts.get("role_definition", prompts.get("system_prompt", ""))
        if not role_def or role_def.startswith("# TODO"):
            warnings.append("Role definition appears to be a placeholder - please customize")

    # Check model
    model = data.get("model", {})
    if not model:
        if config_type != "multi_agent":
            errors.append("Missing 'model' section")
    else:
        model_name = model.get("model_name", "")
        if not model_name or model_name.startswith("# TODO"):
            errors.append("Model name is missing or a placeholder")

        adapter = model.get("adapter", "")
        valid_adapters = ["HuggingFaceAdapter", "OpenAIAdapter", "OllamaAdapter"]
        if adapter and adapter not in valid_adapters:
            errors.append(f"Unknown adapter '{adapter}'. Valid: {valid_adapters}")

    # Check agent section for agent type
    if config_type == "agent":
        agent = data.get("agent", {})
        if not agent:
            warnings.append("Missing 'agent' section - using defaults")
        else:
            tools = agent.get("tools", [])
            if tools and any(str(t).startswith("# TODO") for t in tools):
                warnings.append("Tools list contains placeholders")

    # Check multi-agent specific
    if config_type == "multi_agent":
        if "manager" not in data:
            errors.append("Multi-agent config missing 'manager' section")
        if "workers" not in data:
            errors.append("Multi-agent config missing 'workers' section")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_training_examples(training_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate training examples file.

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Check file exists
    if not Path(training_path).exists():
        return False, [f"Training file not found: {training_path}"], []

    # Try to load JSON
    try:
        with open(training_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in training file: {e}"], []

    # Should be a list
    if not isinstance(data, list):
        return False, ["Training data must be a JSON array of examples"], []

    if len(data) == 0:
        return False, ["Training data is empty - need at least 1 example"], []

    # Check each example
    placeholder_count = 0
    missing_inputs = 0
    missing_outputs = 0
    has_full_trace = 0

    for i, example in enumerate(data):
        if not isinstance(example, dict):
            errors.append(f"Example {i+1}: Must be a JSON object")
            continue

        inputs = example.get("inputs", {})
        if not inputs:
            missing_inputs += 1
        elif not inputs.get("user_input"):
            missing_inputs += 1

        output = example.get("expected_output", "")
        if not output:
            missing_outputs += 1
        elif str(output).startswith("# TODO"):
            placeholder_count += 1

        if example.get("full_trace"):
            has_full_trace += 1

    if missing_inputs > 0:
        errors.append(f"{missing_inputs} example(s) missing 'inputs.user_input'")

    if missing_outputs > 0:
        errors.append(f"{missing_outputs} example(s) missing 'expected_output'")

    if placeholder_count > 0:
        warnings.append(f"{placeholder_count} example(s) have placeholder outputs (# TODO)")

    if has_full_trace > 0 and has_full_trace < len(data):
        warnings.append(
            f"Only {has_full_trace}/{len(data)} examples have full_trace - consider adding to all"
        )

    if len(data) < 5:
        warnings.append(f"Only {len(data)} examples - recommend 10-50 for better optimization")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="fair-optimize",
        description="FAIR Prompt Optimizer - DSPy-powered optimization for FAIR-LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fair-optimize init --type agent
  fair-optimize validate -c config.json -t examples.json
  fair-optimize optimize -c config.json -t examples.json --metric contains_answer
  fair-optimize test -c config_optimized.json -i "What is 2+2?"
  fair-optimize info -c config_optimized.json
        """,
    )

    parser.add_argument("--version", "-V", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- init command ---
    init_parser = subparsers.add_parser("init", help="Initialize a new config file")
    init_parser.add_argument(
        "--type",
        "-t",
        choices=["simple_llm", "agent", "multi_agent"],
        default="agent",
        help="Type of config to create (default: agent)",
    )
    init_parser.add_argument(
        "--output", "-o", default=None, help="Output file path (default: {type}_config.json)"
    )

    # --- optimize command ---
    opt_parser = subparsers.add_parser("optimize", help="Optimize an agent's prompts")
    opt_parser.add_argument("--config", "-c", required=True, help="Path to agent config JSON")
    opt_parser.add_argument(
        "--training", "-t", required=True, help="Path to training examples JSON"
    )
    opt_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for optimized config (default: {config}_optimized.json)",
    )
    opt_parser.add_argument(
        "--optimizer",
        choices=["bootstrap", "mipro"],
        default="bootstrap",
        help="DSPy optimizer to use (default: bootstrap)",
    )
    opt_parser.add_argument(
        "--metric",
        choices=[
            "exact_match",
            "contains_answer",
            "numeric_accuracy",
            "fuzzy_match",
            "json_format_compliance",
            "format_compliance_score",
            "numeric_accuracy_with_format",
            "sentiment_format_metric",
            "research_quality_metric",
        ],
        default="contains_answer",
        help="Evaluation metric (default: contains_answer)",
    )
    opt_parser.add_argument(
        "--max-demos", type=int, default=4, help="Max bootstrapped demos (default: 4)"
    )
    opt_parser.add_argument("--model", default=None, help="Model name to use (overrides config)")
    opt_parser.add_argument(
        "--adapter",
        choices=["HuggingFaceAdapter", "OpenAIAdapter", "OllamaAdapter"],
        default=None,
        help="LLM adapter to use (overrides config)",
    )
    opt_parser.add_argument(
        "--mipro-lm", default=None, help="DSPy LM for MIPROv2 (e.g., 'ollama_chat/llama3:8b')"
    )
    opt_parser.add_argument(
        "--mipro-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 intensity (default: light)",
    )
    opt_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    opt_parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without running optimization"
    )

    # --- test command ---
    test_parser = subparsers.add_parser("test", help="Test an agent interactively")
    test_parser.add_argument("--config", "-c", required=True, help="Path to agent config JSON")
    test_parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Single input to test (if not provided, enters interactive mode)",
    )
    test_parser.add_argument("--model", default=None, help="Model name to use (overrides config)")
    test_parser.add_argument(
        "--adapter", default=None, help="LLM adapter to use (overrides config)"
    )

    # --- info command ---
    info_parser = subparsers.add_parser("info", help="Show config information")
    info_parser.add_argument("--config", "-c", required=True, help="Path to config JSON")
    info_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full config details"
    )

    # --- compare command ---
    compare_parser = subparsers.add_parser("compare", help="Compare two configs")
    compare_parser.add_argument("config1", help="First config file")
    compare_parser.add_argument("config2", help="Second config file")

    # --- examples command ---
    examples_parser = subparsers.add_parser("examples", help="Create example training data")
    examples_parser.add_argument("--output", "-o", default="examples.json", help="Output file path")
    examples_parser.add_argument(
        "--count", "-n", type=int, default=5, help="Number of example templates to create"
    )

    # --- validate command ---
    validate_parser = subparsers.add_parser("validate", help="Validate config and training data")
    validate_parser.add_argument("--config", "-c", required=True, help="Path to config JSON")
    validate_parser.add_argument(
        "--training", "-t", default=None, help="Path to training examples JSON (optional)"
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
        elif args.command == "validate":
            cmd_validate(args)
    except KeyboardInterrupt:
        print("\n" + Colors.YELLOW + "Interrupted." + Colors.RESET)
        sys.exit(1)
    except FileNotFoundError as e:
        print_error(str(e), "Check the file path and try again")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}", "Check your JSON syntax")
        sys.exit(1)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


# =============================================================================
# Command: init
# =============================================================================


def cmd_init(args):
    """Create a new config file from template."""

    output_path = args.output or f"{args.type}_config.json"

    # Check if file already exists
    if Path(output_path).exists():
        print_warning(f"File already exists: {output_path}")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    if args.type == "simple_llm":
        config = create_simple_llm_template()
    elif args.type == "agent":
        config = create_agent_template()
    elif args.type == "multi_agent":
        config = create_multi_agent_template()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print_success(f"Created {args.type} config: {output_path}")
    print()
    print(f"{Colors.BOLD}Next steps:{Colors.RESET}")
    print(f"  1. Edit {Colors.CYAN}{output_path}{Colors.RESET} to configure your agent")
    print(
        f"  2. Create training examples: {Colors.DIM}fair-optimize examples -o examples.json{Colors.RESET}"
    )
    print(
        f"  3. Validate: {Colors.DIM}fair-optimize validate -c {output_path} -t examples.json{Colors.RESET}"
    )
    print(
        f"  4. Optimize: {Colors.DIM}fair-optimize optimize -c {output_path} -t examples.json{Colors.RESET}"
    )


def create_simple_llm_template() -> dict:
    """Create a simple_llm config template."""
    return {
        "version": "1.0",
        "type": "simple_llm",
        "prompts": {
            "role_definition": "# TODO: Define your system prompt\n# Example: You are a helpful classifier. Always respond with: CATEGORY: [label]",
            "tool_instructions": [],
            "worker_instructions": [],
            "format_instructions": ["# TODO: Add format instructions if needed"],
            "examples": [],
        },
        "model": {
            "adapter": "HuggingFaceAdapter",
            "model_name": "# TODO: Set your model name (e.g., dolphin3-qwen25-3b)",
            "adapter_kwargs": {},
        },
        "agent": {"planner_type": "None", "tools": [], "max_steps": 1, "stateless": True},
    }


def create_agent_template() -> dict:
    """Create an agent config template."""
    return {
        "version": "1.0",
        "type": "agent",
        "prompts": {
            "role_definition": "# TODO: Define your agent's role and goal\n# Example: You are a helpful research assistant with access to tools.",
            "tool_instructions": [
                {"name": "# TODO: tool_name", "description": "# TODO: What this tool does"}
            ],
            "worker_instructions": [],
            "format_instructions": [
                "Always show your reasoning in a 'Thought' section.",
                "Provide actions as JSON with 'tool_name' and 'tool_input'.",
                "When you have the final answer, use the 'final_answer' action.",
            ],
            "examples": [],
        },
        "model": {
            "adapter": "HuggingFaceAdapter",
            "model_name": "# TODO: Set your model name",
            "adapter_kwargs": {},
        },
        "agent": {
            "planner_type": "SimpleReActPlanner",
            "tools": ["# TODO: List tool class names (e.g., SafeCalculatorTool)"],
            "max_steps": 10,
            "stateless": False,
        },
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
                        "role_description": "# TODO: What this worker does",
                    },
                    {
                        "name": "# TODO: Worker2",
                        "role_description": "# TODO: What this worker does",
                    },
                ],
                "format_instructions": [
                    'Use \'delegate\' to assign tasks: {"tool_name": "delegate", "tool_input": {"worker_name": "...", "task": "..."}}',
                    'Use \'final_answer\' when complete: {"tool_name": "final_answer", "tool_input": "..."}',
                ],
                "examples": [],
            },
            "model": {"adapter": "HuggingFaceAdapter", "model_name": "# TODO: Set model"},
            "agent": {"planner_type": "ManagerPlanner", "tools": [], "max_steps": 15},
        },
        "workers": {
            "Worker1": {
                "prompts": {
                    "role_definition": "# TODO: Define worker's specialty",
                    "tool_instructions": [],
                    "format_instructions": [],
                    "examples": [],
                },
                "model": {"adapter": "HuggingFaceAdapter", "model_name": "# TODO: Set model"},
                "agent": {
                    "planner_type": "SimpleReActPlanner",
                    "tools": [],
                    "max_steps": 5,
                    "stateless": True,
                },
            }
        },
        "max_delegation_steps": 15,
    }


# =============================================================================
# Command: optimize
# =============================================================================


def cmd_optimize(args):
    """Run optimization on an agent."""
    quiet = getattr(args, "quiet", False)

    # === Step 1: Validate inputs ===
    if not quiet:
        print_step(1, 5, "Validating inputs...")

    # Validate config
    config_valid, config_errors, config_warnings = validate_config(args.config)
    if not config_valid:
        for err in config_errors:
            print_error(err)
        sys.exit(1)
    for warn in config_warnings:
        if not quiet:
            print_warning(warn)

    # Validate training data
    training_valid, training_errors, training_warnings = validate_training_examples(args.training)
    if not training_valid:
        for err in training_errors:
            print_error(err)
        sys.exit(1)
    for warn in training_warnings:
        if not quiet:
            print_warning(warn)

    if not quiet:
        print_success("Inputs validated")

    # Dry run mode - stop here
    if getattr(args, "dry_run", False):
        print_info("Dry run complete - inputs are valid")
        return

    import nest_asyncio

    nest_asyncio.apply()

    from . import metrics as metrics_module
    from .config import (
        load_optimized_config,
        load_training_examples,
        save_optimized_config,
    )
    from .optimizers import AgentOptimizer, MultiAgentOptimizer, SimpleLLMOptimizer

    # === Step 2: Load config ===
    if not quiet:
        print_step(2, 5, f"Loading config: {args.config}")
    config = load_optimized_config(args.config)
    config_type = config.type

    # Load training examples
    examples = load_training_examples(args.training)
    if not quiet:
        print_success(f"Loaded {len(examples)} training examples")

    # === Step 3: Initialize LLM ===
    if not quiet:
        print_step(3, 5, "Initializing LLM...")

    with Spinner("Loading model...") if not quiet else nullcontext():
        llm = create_llm(
            config.config.get("model", {}),
            model_override=args.model,
            adapter_override=args.adapter,
        )

    if not quiet:
        model_name = args.model or config.config.get("model", {}).get("model_name", "unknown")
        print_success(f"LLM initialized: {model_name}")

    # Get metric function
    metric = getattr(metrics_module, args.metric)

    # === Step 4: Create optimizer ===
    if not quiet:
        print_step(4, 5, f"Creating {config_type} optimizer...")

    if config_type == "simple_llm":
        system_prompt = config.prompts.get("role_definition", "You are a helpful assistant.")
        optimizer = SimpleLLMOptimizer(llm, system_prompt)
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
            print_warning("MIPROv2 works best with --mipro-lm specified")

    # === Step 5: Run optimization ===
    if not quiet:
        print_step(5, 5, f"Running {args.optimizer} optimization...")
        print()
        print(f"  {Colors.DIM}Metric: {args.metric}{Colors.RESET}")
        print(f"  {Colors.DIM}Max demos: {args.max_demos}{Colors.RESET}")
        if args.optimizer == "mipro":
            print(f"  {Colors.DIM}MIPROv2 intensity: {args.mipro_auto}{Colors.RESET}")
        print()

    with Spinner("Optimizing prompts...") if not quiet else nullcontext():
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

    # Summary
    if not quiet:
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}  OPTIMIZATION COMPLETE{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print()
        print(f"  Output: {Colors.CYAN}{output_path}{Colors.RESET}")
        print(f"  Examples generated: {Colors.BOLD}{len(result.examples)}{Colors.RESET}")
        print(f"  Optimization runs: {len(result.optimization.runs)}")
        print()
        print(f"{Colors.BOLD}Next steps:{Colors.RESET}")
        print(f"  Test: {Colors.DIM}fair-optimize test -c {output_path}{Colors.RESET}")
        print(f"  Info: {Colors.DIM}fair-optimize info -c {output_path}{Colors.RESET}")
    else:
        print(output_path)  # Just output the path in quiet mode


# =============================================================================
# Command: validate
# =============================================================================


def cmd_validate(args):
    """Validate config and training data without running optimization."""
    print()
    print(f"{Colors.BOLD}Validating files...{Colors.RESET}")
    print()

    all_valid = True

    # Validate config
    print(f"Config: {args.config}")
    config_valid, config_errors, config_warnings = validate_config(args.config)

    if config_valid:
        print_success("Config is valid")
    else:
        print_error("Config has errors")
        all_valid = False

    for err in config_errors:
        print(f"  {Colors.RED}✗{Colors.RESET} {err}")
    for warn in config_warnings:
        print(f"  {Colors.YELLOW}!{Colors.RESET} {warn}")

    # Validate training data if provided
    if args.training:
        print()
        print(f"Training data: {args.training}")
        training_valid, training_errors, training_warnings = validate_training_examples(
            args.training
        )

        if training_valid:
            print_success("Training data is valid")
        else:
            print_error("Training data has errors")
            all_valid = False

        for err in training_errors:
            print(f"  {Colors.RED}✗{Colors.RESET} {err}")
        for warn in training_warnings:
            print(f"  {Colors.YELLOW}!{Colors.RESET} {warn}")

        # Additional: count examples
        try:
            with open(args.training) as f:
                data = json.load(f)
            if isinstance(data, list):
                print_info(f"Total examples: {len(data)}")
                with_trace = sum(1 for ex in data if ex.get("full_trace"))
                if with_trace > 0:
                    print_info(f"Examples with full_trace: {with_trace}")
        except Exception:
            pass

    # Summary
    print()
    if all_valid:
        print(f"{Colors.GREEN}{Colors.BOLD}All validations passed!{Colors.RESET}")
        if args.training:
            print("\nReady to optimize:")
            print(
                f"  {Colors.DIM}fair-optimize optimize -c {args.config} -t {args.training}{Colors.RESET}"
            )
    else:
        print(f"{Colors.RED}{Colors.BOLD}Validation failed - fix errors above{Colors.RESET}")
        sys.exit(1)


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

        # Check both system_prompt and role_definition for backwards compatibility
        system_prompt = config.prompts.get(
            "system_prompt", config.prompts.get("role_definition", "")
        )
        tester = SimpleLLMOptimizer(llm, system_prompt, config=config)

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

            if user_input.lower() in ["quit", "exit", "q"]:
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
    print("\nModel:")
    print(f"  Adapter: {model.get('adapter', 'unknown')}")
    print(f"  Name: {model.get('model_name', 'unknown')}")

    # Prompts
    prompts = config.prompts
    print("\nPrompts:")
    print(f"  Role definition: {len(prompts.get('role_definition', '') or '')} chars")
    print(f"  Tool instructions: {len(prompts.get('tool_instructions', []))}")
    print(f"  Worker instructions: {len(prompts.get('worker_instructions', []))}")
    print(f"  Format instructions: {len(prompts.get('format_instructions', []))}")
    print(f"  Examples: {len(prompts.get('examples', []))}")

    # Agent spec
    agent = config.config.get("agent", {})
    if agent:
        print("\nAgent:")
        print(f"  Planner: {agent.get('planner_type', 'unknown')}")
        print(f"  Tools: {agent.get('tools', [])}")
        print(f"  Max steps: {agent.get('max_steps', 'unknown')}")

    # Optimization provenance
    opt = config.optimization
    print("\nOptimization:")
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
            print("\n  Run history:")
            for i, run in enumerate(opt.runs, 1):
                print(
                    f"    {i}. {run.optimizer} ({run.timestamp[:10]}) - {run.examples_before}→{run.examples_after} examples"
                )

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
        print("\n[CHANGED] Role definition:")
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
        examples.append(
            {
                "inputs": {"user_input": f"# TODO: Example {i+1} input"},
                "expected_output": f"# TODO: Example {i+1} expected output",
            }
        )

    # Add helpful comments

    # Save as just the examples array (without metadata for actual use)
    with open(args.output, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Created training examples template: {args.output}")
    print(f"  {args.count} example templates created")
    print()
    print("Edit the file to add your actual training examples.")
    print("Each example needs:")
    print("  - inputs.user_input: The input prompt")
    print("  - expected_output: What the agent should produce")


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
