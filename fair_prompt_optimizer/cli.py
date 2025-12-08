# cli.py

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    FAIR Prompt Optimizer - Optimize FAIR-LLM agent prompts using DSPy.
    """
    pass


@main.command()
def optimize():
    """
    Run prompt optimization.
    
    Note: Full agent optimization requires the Python API.
    """
    console.print(Panel.fit(
        "[bold yellow]CLI optimization is not supported.[/bold yellow]\n\n"
        "The optimizer requires a FAIR-LLM agent instance, which cannot be\n"
        "passed via command line.\n\n"
        "[bold green]Use the Python API instead:[/bold green]\n\n"
        "[dim]from fairlib import SimpleAgent, SimpleReActPlanner, ...\n"
        "from fair_prompt_optimizer import FAIRPromptOptimizer, TrainingExample\n\n"
        "# Build your agent\n"
        "agent = SimpleAgent(llm, planner, executor, memory)\n\n"
        "# Optimize\n"
        "optimizer = FAIRPromptOptimizer(agent)\n"
        "config = optimizer.compile(\n"
        "    training_examples=examples,\n"
        "    metric=my_metric,\n"
        ")[/dim]\n\n"
        "See: [link=https://github.com/USAFA-AI-Center/fair_prompt_optimizer]"
        "examples/optimize_fair_agent.py[/link]",
        title="FAIR Prompt Optimizer",
        border_style="blue"
    ))


@main.command()
def info():
    """
    Show information about the optimizer.
    """
    console.print(Panel.fit(
        f"[bold]FAIR Prompt Optimizer[/bold] v1.0.0\n\n"
        "DSPy-powered prompt optimization for FAIR-LLM agents.\n\n"
        "[bold]Features:[/bold]\n"
        "• BootstrapFewShot optimization\n"
        "• MIPROv2 optimization\n"
        "• Full agent execution during optimization\n"
        "• Automatic config extraction and application\n\n"
        "[bold]Usage:[/bold]\n"
        "Use the Python API to optimize your FAIR-LLM agent's prompts.\n"
        "See examples/optimize_fair_agent.py for a complete example.",
        title="About",
        border_style="green"
    ))


if __name__ == "__main__":
    main()