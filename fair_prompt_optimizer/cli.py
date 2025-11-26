"""
cli.py
======

Command-line interface for fair_prompt_optimizer.

Provides a simple way to run prompt optimization from the terminal:

    fair-optimize --input base.json --output optimized.json --training-data examples.json

"""

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .optimizers import FAIRPromptOptimizer
from .translator import load_fair_config, load_training_examples
from .metrics import exact_match, contains_answer, numeric_accuracy, fuzzy_match

# Setup rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


# Available metrics for CLI
METRICS = {
    "exact": exact_match,
    "contains": contains_answer,
    "numeric": numeric_accuracy,
    "fuzzy": fuzzy_match,
}


@click.group()
@click.version_option(version=__version__)
def main():
    """
    FAIR Prompt Optimizer - Optimize FAIR-LLM prompts using DSPy.
    
    This tool takes FAIR-LLM prompt configurations and optimizes them
    using DSPy's BootstrapFewShot or MIPROv2 algorithms.
    """
    pass


@main.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the FAIR-LLM JSON configuration to optimize"
)
@click.option(
    "--output", "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Path to save the optimized configuration"
)
@click.option(
    "--training-data", "-t",
    "training_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSON file with training examples"
)
@click.option(
    "--optimizer",
    type=click.Choice(["bootstrap", "bootstrap-rs", "mipro-light", "mipro-medium", "mipro-heavy"]),
    default="bootstrap",
    help="Optimization algorithm to use"
)
@click.option(
    "--model", "-m",
    default="openai/gpt-4o-mini",
    help="LLM model to use for optimization"
)
@click.option(
    "--metric",
    type=click.Choice(list(METRICS.keys())),
    default="exact",
    help="Evaluation metric to use"
)
@click.option(
    "--max-demos",
    default=4,
    type=int,
    help="Maximum number of demos per category"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
def optimize(
    input_path: str,
    output_path: str,
    training_data_path: str,
    optimizer: str,
    model: str,
    metric: str,
    max_demos: int,
    verbose: bool
):
    """
    Optimize a FAIR-LLM prompt configuration.
    
    Example:
    
        fair-optimize -i base.json -o optimized.json -t training.json --optimizer mipro-light
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        f"[bold blue]FAIR Prompt Optimizer v{__version__}[/bold blue]",
        subtitle="Powered by DSPy"
    ))
    
    # Display configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Input", input_path)
    config_table.add_row("Output", output_path)
    config_table.add_row("Training Data", training_data_path)
    config_table.add_row("Optimizer", optimizer)
    config_table.add_row("Model", model)
    config_table.add_row("Metric", metric)
    config_table.add_row("Max Demos", str(max_demos))
    console.print(config_table)
    console.print()
    
    # Load training data to show count
    training_examples = load_training_examples(training_data_path)
    console.print(f"Loaded [bold]{len(training_examples)}[/bold] training examples")
    
    # Get the metric function
    metric_fn = METRICS[metric]
    
    # Create optimizer
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Initializing optimizer...", total=None)
        opt = FAIRPromptOptimizer(lm_model=model)
    
    console.print(f"Initialized optimizer with model: [bold]{model}[/bold]")
    console.print()
    
    # Run optimization
    console.print(f"[bold]Running {optimizer} optimization...[/bold]")
    console.print("  This may take a few minutes depending on your configuration.")
    console.print()
    
    try:
        if optimizer == "bootstrap":
            result = opt.optimize_bootstrap(
                fair_config_path=input_path,
                training_examples=training_examples,
                metric=metric_fn,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
                output_path=output_path
            )
        elif optimizer == "bootstrap-rs":
            result = opt.optimize_bootstrap(
                fair_config_path=input_path,
                training_examples=training_examples,
                metric=metric_fn,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
                output_path=output_path,
                use_random_search=True
            )
        elif optimizer.startswith("mipro"):
            auto = optimizer.split("-")[1]  # light, medium, or heavy
            result = opt.optimize_mipro(
                fair_config_path=input_path,
                training_examples=training_examples,
                metric=metric_fn,
                auto=auto,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
                output_path=output_path
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Display results
        console.print()
        console.print(Panel.fit(
            f"[bold green]Optimization Complete![/bold green]",
        ))
        
        results_table = Table(title="Results", show_header=False)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_row("Optimizer", result.optimizer_name)
        results_table.add_row("Duration", f"{result.duration_seconds:.2f}s")
        results_table.add_row("Output", result.output_path)
        results_table.add_row("Examples Generated", str(len(result.config.examples)))
        console.print(results_table)
        
        console.print()
        console.print(f"Optimized configuration saved to: [bold]{output_path}[/bold]")
        
    except Exception as e:
        console.print(f"[bold red]Optimization failed:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option(
    "--config", "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to FAIR-LLM JSON configuration to evaluate"
)
@click.option(
    "--test-data", "-t",
    "test_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSON file with test examples"
)
@click.option(
    "--model", "-m",
    default="openai/gpt-4o-mini",
    help="LLM model to use for evaluation"
)
@click.option(
    "--metric",
    type=click.Choice(list(METRICS.keys())),
    default="exact",
    help="Evaluation metric to use"
)
def evaluate(
    config_path: str,
    test_data_path: str,
    model: str,
    metric: str
):
    """
    Evaluate a FAIR-LLM configuration on test data.
    
    Useful for comparing before/after optimization performance.
    
    Example:
    
        fair-optimize evaluate -c optimized.json -t test.json
    """
    console.print(Panel.fit(
        f"[bold blue]FAIR Prompt Evaluator[/bold blue]"
    ))
    
    # Load config and test data
    config = load_fair_config(config_path)
    test_examples = load_training_examples(test_data_path)
    
    console.print(f"Config: [bold]{config_path}[/bold]")
    console.print(f"   Optimized: {'✅' if config.metadata.optimized else '❌'}")
    console.print(f"   Test examples: [bold]{len(test_examples)}[/bold]")
    console.print()
    
    # Create optimizer and run evaluation
    opt = FAIRPromptOptimizer(lm_model=model)
    metric_fn = METRICS[metric]
    
    console.print("Running evaluation...")
    result = opt.evaluate(
        fair_config_path=config_path,
        test_examples=test_examples,
        metric=metric_fn
    )
    
    # Display results
    console.print()
    results_table = Table(title="Evaluation Results", show_header=False)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Score", f"{result['score']:.2%}")
    results_table.add_row("Test Examples", str(result['num_examples']))
    results_table.add_row("Config Optimized", "Yes" if result['optimized'] else "No")
    console.print(results_table)


@main.command()
@click.option(
    "--config", "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to FAIR-LLM JSON configuration to inspect"
)
def inspect(config_path: str):
    """
    Inspect a FAIR-LLM configuration file.
    
    Displays the configuration contents in a readable format.
    """
    config = load_fair_config(config_path)
    
    console.print(Panel.fit(
        f"[bold blue]Configuration: {config_path}[/bold blue]"
    ))
    
    # Basic info
    info_table = Table(title="Metadata", show_header=False)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Version", config.version)
    info_table.add_row("Optimized", "✅ Yes" if config.metadata.optimized else "❌ No")
    if config.metadata.optimized:
        info_table.add_row("Optimizer", config.metadata.optimizer or "Unknown")
        info_table.add_row("Optimized At", config.metadata.optimized_at or "Unknown")
        if config.metadata.score:
            info_table.add_row("Score", f"{config.metadata.score:.2%}")
    console.print(info_table)
    console.print()
    
    # Role definition
    if config.role_definition:
        console.print(Panel(
            config.role_definition[:500] + ("..." if len(config.role_definition) > 500 else ""),
            title="Role Definition",
            border_style="blue"
        ))
        console.print()
    
    # Tools
    if config.tool_instructions:
        tools_table = Table(title="Tool Instructions")
        tools_table.add_column("Name", style="cyan")
        tools_table.add_column("Description", style="white")
        for tool in config.tool_instructions:
            tools_table.add_row(tool.name, tool.description[:60] + "...")
        console.print(tools_table)
        console.print()
    
    # Examples
    if config.examples:
        console.print(f"[bold]Examples:[/bold] {len(config.examples)} total")
        for i, example in enumerate(config.examples[:3], 1):
            console.print(Panel(
                example[:300] + ("..." if len(example) > 300 else ""),
                title=f"Example {i}",
                border_style="green"
            ))
        if len(config.examples) > 3:
            console.print(f"   ... and {len(config.examples) - 3} more")


@main.command()
def init():
    """
    Create a sample configuration and training data file.
    
    Generates starter files to help you get started with optimization.
    """
    # Sample config
    sample_config = {
        "version": "1.0",
        "role_definition": "You are a helpful AI assistant that answers questions accurately and concisely.",
        "tool_instructions": [],
        "worker_instructions": [],
        "format_instructions": [
            "Provide clear, direct answers",
            "If uncertain, acknowledge the uncertainty"
        ],
        "examples": [],
        "metadata": {
            "optimized": False
        }
    }
    
    # Sample training data
    sample_training = [
        {
            "inputs": {"user_query": "What is 2 + 2?"},
            "expected_output": "4"
        },
        {
            "inputs": {"user_query": "What is the capital of France?"},
            "expected_output": "Paris"
        },
        {
            "inputs": {"user_query": "Who wrote Romeo and Juliet?"},
            "expected_output": "William Shakespeare"
        }
    ]
    
    # Write files
    config_path = Path("sample_config.json")
    training_path = Path("sample_training.json")
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    with open(training_path, 'w') as f:
        json.dump(sample_training, f, indent=2)
    
    console.print(Panel.fit(
        "[bold green]Sample files created![/bold green]"
    ))
    console.print(f"Config: [bold]{config_path}[/bold]")
    console.print(f"Training data: [bold]{training_path}[/bold]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Edit sample_config.json with your role definition")
    console.print("  2. Add more examples to sample_training.json")
    console.print("  3. Run: [bold]fair-optimize -i sample_config.json -o optimized.json -t sample_training.json[/bold]")


if __name__ == "__main__":
    main()
