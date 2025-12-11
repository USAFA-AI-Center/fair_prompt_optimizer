# cli.py
"""
Usage:
    # Using BootstrapFewShot (no external LM needed)
    fair-optimize --config agent.json --training-data examples.json --optimizer bootstrap
    
    # Using MIPROv2 with Ollama
    fair-optimize --config agent.json --training-data examples.json --optimizer mipro --ollama-model llama3:8b
    
    # Using MIPROv2 with OpenAI
    fair-optimize --config agent.json --training-data examples.json --optimizer mipro --openai-model gpt-4o-mini
"""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .translator import load_fair_config, save_fair_config, load_training_examples
from .registry import create_agent_from_config
from .fair_agent_module import FAIRPromptOptimizer
from .metrics import exact_match, contains_answer, numeric_accuracy, fuzzy_match

console = Console()
logger = logging.getLogger(__name__)

# Available metrics
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
    FAIR Prompt Optimizer - Optimize FAIR-LLM agent prompts using DSPy.
    
    Optimize your agent's prompts using BootstrapFewShot or MIPROv2 algorithms.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the FAIR agent config JSON"
)
@click.option(
    "--training-data", "-t",
    "training_data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training examples JSON"
)
@click.option(
    "--output", "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output path for optimized config (default: <input>_optimized.json)"
)
@click.option(
    "--optimizer",
    type=click.Choice(["bootstrap", "mipro"]),
    default="bootstrap",
    help="Optimization algorithm to use"
)
@click.option(
    "--metric",
    type=click.Choice(list(METRICS.keys())),
    default="numeric",
    help="Evaluation metric"
)
@click.option(
    "--max-demos",
    default=4,
    type=int,
    help="Maximum number of demos to generate"
)
@click.option(
    "--mipro-mode",
    type=click.Choice(["light", "medium", "heavy"]),
    default="light",
    help="MIPROv2 optimization intensity"
)
@click.option(
    "--ollama-model",
    default=None,
    help="Ollama model for MIPROv2 (e.g., 'llama3:8b', 'dolphin-llama3:8b')"
)
@click.option(
    "--ollama-url",
    default="http://localhost:11434",
    help="Ollama server URL"
)
@click.option(
    "--openai-model",
    default=None,
    help="OpenAI model for MIPROv2 (e.g., 'gpt-4o-mini')"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
def optimize(
    config_path,
    training_data_path,
    output_path,
    optimizer,
    metric,
    max_demos,
    mipro_mode,
    ollama_model,
    ollama_url,
    openai_model,
    verbose
):
    """
    Optimize an agent's prompts using DSPy algorithms.
    
    Examples:
    
        # BootstrapFewShot (no external LM needed)
        fair-optimize -c agent.json -t examples.json --optimizer bootstrap
        
        # MIPROv2 with Ollama
        fair-optimize -c agent.json -t examples.json --optimizer mipro --ollama-model llama3:8b
        
        # MIPROv2 with OpenAI
        fair-optimize -c agent.json -t examples.json --optimizer mipro --openai-model gpt-4o-mini
    """
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Default output path
    if output_path is None:
        input_stem = Path(config_path).stem
        output_path = f"{input_stem}_optimized.json"
    
    console.print(Panel.fit(
        f"[bold blue]FAIR Prompt Optimizer[/bold blue]\n"
        f"Config: {config_path}\n"
        f"Training data: {training_data_path}\n"
        f"Optimizer: {optimizer}\n"
        f"Output: {output_path}",
        title="Starting Optimization",
        border_style="blue"
    ))
    
    # Validate MIPROv2 requirements
    if optimizer == "mipro" and not ollama_model and not openai_model:
        console.print(Panel.fit(
            "[bold red]MIPROv2 requires an LM for instruction generation.[/bold red]\n\n"
            "Choose one of:\n"
            "  --ollama-model llama3:8b     (local, requires Ollama running)\n"
            "  --openai-model gpt-4o-mini   (API, requires OPENAI_API_KEY)\n\n"
            "Or use --optimizer bootstrap (no external LM needed)",
            title="Missing LM Configuration",
            border_style="red"
        ))
        sys.exit(1)
    
    try:
        # Step 1: Load config and create agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Loading config...", total=None)
            config = load_fair_config(config_path)
            progress.update(task, description="✓ Config loaded")
            
            task = progress.add_task("Loading training examples...", total=None)
            examples = load_training_examples(training_data_path)
            progress.update(task, description=f"✓ Loaded {len(examples)} training examples")
            
            task = progress.add_task("Creating agent from config...", total=None)
            agent = create_agent_from_config(config)
            progress.update(task, description="✓ Agent created")
            
            # Configure DSPy LM if needed for MIPROv2
            dspy_lm = None
            if optimizer == "mipro":
                task = progress.add_task("Configuring DSPy LM...", total=None)
                dspy_lm = _create_dspy_lm(ollama_model, ollama_url, openai_model)
                progress.update(task, description=f"✓ DSPy LM configured")
        
        # Step 2: Run optimization
        console.print(f"\n[bold]Running {optimizer} optimization...[/bold]\n")
        
        optimizer_instance = FAIRPromptOptimizer(agent)
        
        optimized_config = optimizer_instance.compile(
            training_examples=examples,
            metric=METRICS[metric],
            optimizer=optimizer,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            mipro_auto=mipro_mode,
            output_path=output_path,
            dspy_lm=dspy_lm,
        )
        
        # Step 3: Report results
        _display_results(optimized_config, output_path)
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _create_dspy_lm(ollama_model, ollama_url, openai_model):
    """Create DSPy LM based on CLI options."""
    import dspy
    
    if ollama_model:
        console.print(f"  Using Ollama model: {ollama_model} at {ollama_url}")
        return dspy.LM(
            model=f"ollama_chat/{ollama_model}",
            api_base=ollama_url,
        )
    elif openai_model:
        console.print(f"  Using OpenAI model: {openai_model}")
        return dspy.LM(model=f"openai/{openai_model}")
    
    return None


def _display_results(config, output_path):
    """Display optimization results."""
    console.print("\n")
    
    # Results table
    table = Table(title="Optimization Complete", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Output file", output_path)
    table.add_row("Role definition", f"{config.role_definition[:60]}..." if config.role_definition and len(config.role_definition) > 60 else config.role_definition or "N/A")
    table.add_row("Demos generated", str(len(config.examples)))
    table.add_row("Model", f"{config.model.adapter}({config.model.model_name})")
    table.add_row("Tools", ", ".join(config.agent.tools) or "None")
    
    if config.metadata:
        table.add_row("Optimizer", config.metadata.optimizer or "N/A")
        table.add_row("Optimized at", config.metadata.optimized_at or "N/A")
    
    console.print(table)
    
    # Show demos if any
    if config.examples:
        console.print("\n[bold]Generated Demos:[/bold]")
        for i, demo in enumerate(config.examples[:3], 1):  # Show first 3
            lines = demo.split('\n')
            preview = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
            console.print(f"  {i}. {preview}")
        if len(config.examples) > 3:
            console.print(f"  ... and {len(config.examples) - 3} more")
    
    console.print(f"\n[bold green]✓ Optimized config saved to: {output_path}[/bold green]")
    console.print(f"\n[dim]Load optimized agent with:[/dim]")
    console.print(f"  from fair_prompt_optimizer import load_optimized_agent")
    console.print(f"  agent = load_optimized_agent('{output_path}')")


@main.command()
@click.option(
    "--config", "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to an agent config JSON to test"
)
def test(config_path):
    """
    Test an agent config interactively.
    
    Loads an agent from config and runs an interactive chat session.
    """
    import asyncio
    
    console.print(f"Loading agent from: {config_path}")
    
    try:
        config = load_fair_config(config_path)
        agent = create_agent_from_config(config)
        
        console.print(Panel.fit(
            f"[bold green]Agent loaded successfully![/bold green]\n\n"
            f"Model: {config.model.adapter}({config.model.model_name})\n"
            f"Tools: {', '.join(config.agent.tools) or 'None'}\n"
            f"Max steps: {config.agent.max_steps}",
            title="Agent Ready",
            border_style="green"
        ))
        
        console.print("\nType 'exit' to quit.\n")
        
        async def run_loop():
            while True:
                try:
                    user_input = console.input("[bold blue]You:[/bold blue] ")
                    if user_input.lower() in ["exit", "quit"]:
                        console.print("[bold]Goodbye![/bold]")
                        break
                    
                    response = await agent.arun(user_input)
                    console.print(f"[bold green]Agent:[/bold green] {response}\n")
                    
                except KeyboardInterrupt:
                    console.print("\n[bold]Exiting...[/bold]")
                    break
        
        asyncio.run(run_loop())
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@main.command()
def init():
    """
    Create a starter agent config file.
    
    Generates a template config.json that you can customize.
    """
    from .translator import FAIRConfig, ModelConfig, AgentConfig
    
    config = FAIRConfig(
        role_definition=(
            "You are a helpful assistant. You reason step-by-step to solve problems. "
            "If a user's request requires tools, use them appropriately."
        ),
        model=ModelConfig(
            model_name="dolphin3-qwen25-3b",
            adapter="HuggingFaceAdapter",
        ),
        agent=AgentConfig(
            agent_type="SimpleAgent",
            planner_type="SimpleReActPlanner",
            max_steps=10,
            tools=["SafeCalculatorTool"],
        ),
    )
    
    output_path = "agent_config.json"
    save_fair_config(config, output_path)
    
    console.print(Panel.fit(
        f"[bold green]Created starter config:[/bold green] {output_path}\n\n"
        "Edit this file to customize your agent, then run:\n"
        f"  fair-optimize -c {output_path} -t examples.json",
        title="Config Created",
        border_style="green"
    ))
    
    # Also create example training data
    examples_path = "examples.json"
    example_data = [
        {"inputs": {"user_input": "What is 15 + 27?"}, "expected_output": "42"},
        {"inputs": {"user_input": "Calculate 100 divided by 4"}, "expected_output": "25"},
        {"inputs": {"user_input": "What is 8 times 7?"}, "expected_output": "56"},
    ]
    
    import json
    with open(examples_path, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    console.print(f"[dim]Also created example training data: {examples_path}[/dim]")


@main.command()
def info():
    """
    Show information about the optimizer.
    """
    console.print(Panel.fit(
        f"[bold]FAIR Prompt Optimizer[/bold] v{__version__}\n\n"
        "DSPy-powered prompt optimization for FAIR-LLM agents.\n\n"
        "[bold]Optimizers:[/bold]\n"
        "  • bootstrap  - BootstrapFewShot (quick, demo-based)\n"
        "  • mipro      - MIPROv2 (instruction optimization)\n\n"
        "[bold]Metrics:[/bold]\n"
        "  • exact      - Exact string match\n"
        "  • contains   - Answer contained in response\n"
        "  • numeric    - Numeric comparison with tolerance\n"
        "  • fuzzy      - Fuzzy string matching\n\n"
        "[bold]Quick Start:[/bold]\n"
        "  1. fair-optimize init              # Create starter config\n"
        "  2. Edit agent_config.json          # Customize your agent\n"
        "  3. Edit examples.json              # Add training examples\n"
        "  4. fair-optimize -c agent_config.json -t examples.json\n\n"
        "[bold]MIPROv2 with Ollama:[/bold]\n"
        "  ollama serve                       # Start Ollama\n"
        "  ollama pull llama3:8b              # Pull model\n"
        "  fair-optimize -c config.json -t examples.json \\\n"
        "      --optimizer mipro --ollama-model llama3:8b",
        title="About",
        border_style="blue"
    ))


if __name__ == "__main__":
    main()