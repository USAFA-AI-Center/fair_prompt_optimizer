#!/usr/bin/env python3
"""
FAIR Prompt Optimizer - Complete Examples

Demonstrates all optimization workflows:
1. example_optimize_agent()        - Optimize a single agent with full-trace examples
2. example_optimize_simple_llm()   - Optimize a base LLM (no agent/tools)
3. example_optimize_multi_agent()  - Optimize a multi-agent system
4. example_iterative_optimization() - Multiple rounds of optimization
5. example_compare_agents()        - Compare optimized vs non-optimized
6. example_mipro_optimization()    - Use MIPROv2 for instruction optimization

All examples use:
- ReActPlanner (JSON format) for structured output
- Full-trace examples loaded from training_data/math_examples.json

"""

# suppress warnings from uneeded packages
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import asyncio
import logging
from pathlib import Path
import dspy

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Path to training data
TRAINING_DATA_PATH = Path(__file__).parent / "training_data" / "math_examples.json"


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
        load_training_examples,
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

Observation: The result of '15 + 27' is 42

{
    "thought": "The calculator returned 42. I will provide this as the final answer.",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "42"
    }
}'''
    
    prompt_builder = PromptBuilder()
    prompt_builder.role_definition = RoleDefinition(
        "You are a precise math assistant. You MUST use the safe_calculator tool "
        "for ALL calculations. Never compute answers in your head."
    )
    prompt_builder.format_instructions = [
        FormatInstruction(
            "# --- RESPONSE FORMAT ---\n"
            "You MUST respond with a JSON object containing 'thought' and 'action' keys.\n"
            "The 'action' object must have 'tool_name' and 'tool_input' keys.\n\n"
            "Example:\n"
            '{"thought": "I need to calculate X", "action": {"tool_name": "safe_calculator", "tool_input": "2 + 2"}}'
        ),
        FormatInstruction(
            "# --- FINAL ANSWER FORMAT ---\n"
            "When providing the final answer, use tool_name 'final_answer' and "
            "set tool_input to ONLY the numeric result (e.g., '42', not 'The answer is 42')."
        ),
    ]
    prompt_builder.examples = [Example(SEED_EXAMPLE)]  # One seed example
    
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
        mipro_auto='medium',
        training_data_path=str(TRAINING_DATA_PATH),
    )
    
    # Save optimized config
    result.save("agent_optimized.json")
    
    print()
    print(f"Optimized agent: {len(result.examples)} examples")
    print(f"Saved to: agent_optimized.json")
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
        TrainingExample,
        format_compliance,
    )
    
    print("=" * 60)
    print("EXAMPLE 2: OPTIMIZE SIMPLE LLM (No Agent)")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # System prompt for sentiment classification
    system_prompt = """You are a sentiment classifier.

# --- RESPONSE FORMAT (MANDATORY) ---
Respond with ONLY one of:
SENTIMENT: positive
SENTIMENT: negative
SENTIMENT: neutral

No explanations, just the label."""

    # Training examples (no full_trace needed for simple LLM)
    training_examples = [
        TrainingExample(
            inputs={"user_input": "I love this product!"},
            expected_output="SENTIMENT: positive"
        ),
        TrainingExample(
            inputs={"user_input": "This is terrible."},
            expected_output="SENTIMENT: negative"
        ),
        TrainingExample(
            inputs={"user_input": "It's okay."},
            expected_output="SENTIMENT: neutral"
        ),
        TrainingExample(
            inputs={"user_input": "Best purchase ever!"},
            expected_output="SENTIMENT: positive"
        ),
        TrainingExample(
            inputs={"user_input": "Waste of money."},
            expected_output="SENTIMENT: negative"
        ),
    ]
    
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Training examples: {len(training_examples)}")
    print()
    
    optimizer = SimpleLLMOptimizer(llm=llm, system_prompt=system_prompt)
    
    result = optimizer.compile(
        training_examples=training_examples,
        metric=format_compliance("SENTIMENT:"),
        optimizer="bootstrap",
        max_bootstrapped_demos=3,
    )
    
    result.save("classifier_optimized.json")
    
    print(f"Optimized examples: {len(result.prompts.get('examples', []))}")
    print(f"Saved to: classifier_optimized.json")
    print()
    
    # Test
    print("Testing optimized classifier:")
    print("-" * 40)
    for text in ["Amazing!", "Horrible experience.", "It works."]:
        response = optimizer.test(text)
        print(f"Input: {text}")
        print(f"Output: {response}")
        print()
    
    return optimizer


# =============================================================================
# Example 3: Optimize a Multi-Agent System
# =============================================================================

async def example_optimize_multi_agent():
    """
    Optimize a multi-agent system with manager and workers.
    
    Use case: complex tasks requiring coordination between specialized agents.
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
    )
    from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
    from fairlib.modules.orchestration.hierarchical_runner import HierarchicalAgentRunner
    from fairlib.utils.config_manager import save_multi_agent_config, load_multi_agent
    
    from fair_prompt_optimizer import (
        MultiAgentOptimizer,
        load_training_examples,
        contains_answer,
    )
    
    print("=" * 60)
    print("EXAMPLE 3: OPTIMIZE MULTI-AGENT SYSTEM")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    # -------------------------------------------------------------------------
    # Create Calculator Worker
    # -------------------------------------------------------------------------
    calc_registry = ToolRegistry()
    calc_registry.register_tool(SafeCalculatorTool())
    
    calc_builder = PromptBuilder()
    calc_builder.role_definition = RoleDefinition(
        "You are a calculator specialist. Use safe_calculator for all math."
    )
    calc_builder.format_instructions = [
        FormatInstruction(
            "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"...\", \"tool_input\": \"...\"}}"
        ),
    ]
    calc_builder.examples = []
    
    calc_planner = ReActPlanner(llm, calc_registry, prompt_builder=calc_builder)
    calculator_worker = SimpleAgent(
        llm=llm,
        planner=calc_planner,
        tool_executor=ToolExecutor(calc_registry),
        memory=WorkingMemory(),
        max_steps=5,
    )
    
    # -------------------------------------------------------------------------
    # Create Manager
    # -------------------------------------------------------------------------
    manager_registry = ToolRegistry()
    
    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are a manager that delegates math problems to the Calculator worker."
    )
    manager_builder.format_instructions = [
        FormatInstruction(
            "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"Calculator\", \"tool_input\": \"task\"}}"
        ),
    ]
    manager_builder.worker_instructions = [
        WorkerInstruction(name="Calculator", role_description="Handles math calculations."),
    ]
    manager_builder.examples = []
    
    manager_planner = ReActPlanner(llm, manager_registry, prompt_builder=manager_builder)
    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=ToolExecutor(manager_registry),
        memory=WorkingMemory(),
        max_steps=5,
    )
    
    # -------------------------------------------------------------------------
    # Create Runner and Optimize
    # -------------------------------------------------------------------------
    runner = HierarchicalAgentRunner(
        manager=manager_agent,
        workers={"Calculator": calculator_worker},
    )
    
    save_multi_agent_config(runner, "multi_agent_initial.json")
    print(f"Initial multi-agent system saved")
    
    training_examples = load_training_examples(str(TRAINING_DATA_PATH))
    
    optimizer = MultiAgentOptimizer(runner)
    
    result = optimizer.compile(
        training_examples=training_examples[:4],
        metric=contains_answer,
        optimizer="bootstrap",
        optimize_manager=True,
        optimize_workers=False,
        max_bootstrapped_demos=2,
    )
    
    result.save("multi_agent_optimized.json")
    
    print(f"Optimized multi-agent system saved")
    print(f"Optimization runs: {len(result.optimization.runs)}")
    print()
    
    # Test
    print("Testing optimized multi-agent system:")
    print("-" * 40)
    
    optimized_runner = load_multi_agent("multi_agent_optimized.json", llm)
    test_result = await optimized_runner.run("What is 50% of 80?")
    print(f"Q: What is 50% of 80?")
    print(f"A: {test_result}")
    
    return optimizer


# =============================================================================
# Example 4: Iterative Optimization
# =============================================================================

async def example_iterative_optimization():
    """
    Run multiple rounds of optimization, accumulating examples over time.
    
    This shows how to build up a library of examples iteratively.
    """
    from fairlib import HuggingFaceAdapter
    from fairlib.utils.config_manager import load_agent
    
    from fair_prompt_optimizer import (
        AgentOptimizer,
        TrainingExample,
        load_optimized_config,
        numeric_accuracy,
    )
    
    print("=" * 60)
    print("EXAMPLE 4: ITERATIVE OPTIMIZATION")
    print("=" * 60)
    print()
    
    # Requires agent_optimized.json from example 1
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    try:
        agent = load_agent("agent_optimized.json", llm)
        config = load_optimized_config("agent_optimized.json")
    except FileNotFoundError:
        print("ERROR: Run example_optimize_agent() first to create agent_optimized.json")
        return None
    
    print(f"Loaded agent with {len(config.examples)} examples")
    print(f"Previous optimization runs: {len(config.optimization.runs)}")
    print()
    
    # Round 2: Add more training data
    round2_examples = [
        TrainingExample(
            inputs={"user_input": "What is 100 divided by 4?"},
            expected_output="25",
            full_trace="# --- Example: Division ---\n\"What is 100 divided by 4?\"\n\n{\n    \"thought\": \"I need to divide 100 by 4.\",\n    \"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"100 / 4\"}\n}\n\nObservation: The result of '100 / 4' is 25.0\n\n{\n    \"thought\": \"The result is 25.\",\n    \"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"100 / 4 = 25\"}\n}"
        ),
        TrainingExample(
            inputs={"user_input": "What is 9 squared?"},
            expected_output="81",
            full_trace="# --- Example: Square ---\n\"What is 9 squared?\"\n\n{\n    \"thought\": \"I need to calculate 9^2.\",\n    \"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"9 ** 2\"}\n}\n\nObservation: The result of '9 ** 2' is 81\n\n{\n    \"thought\": \"9 squared is 81.\",\n    \"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"9 squared is 81\"}\n}"
        ),
    ]
    
    print(f"Round 2 training examples: {len(round2_examples)}")
    
    # Optimize with existing config (preserves history)
    optimizer = AgentOptimizer(agent, config=config)
    
    result = optimizer.compile(
        training_examples=round2_examples,
        metric=numeric_accuracy,
        optimizer="bootstrap",
        max_bootstrapped_demos=2,
    )
    
    result.save("agent_optimized_v2.json")
    
    print()
    print(f"After round 2: {len(result.examples)} examples")
    print(f"Total optimization runs: {len(result.optimization.runs)}")
    print(f"Saved to: agent_optimized_v2.json")
    
    return optimizer


# =============================================================================
# Example 5: Compare Optimized vs Non-Optimized Agent
# =============================================================================

async def example_compare_agents():
    """
    Compare the performance of an agent before and after optimization.
    
    This demonstrates the value of few-shot examples.
    """
    from fairlib import HuggingFaceAdapter
    from fairlib.utils.config_manager import load_agent
    
    print("=" * 60)
    print("EXAMPLE 5: COMPARE OPTIMIZED vs NON-OPTIMIZED")
    print("=" * 60)
    print()
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    try:
        initial_agent = load_agent("agent_initial.json", llm)
        optimized_agent = load_agent("agent_optimized.json", llm)
    except FileNotFoundError:
        print("ERROR: Run example_optimize_agent() first")
        return
    
    print(f"Initial agent: {len(initial_agent.planner.prompt_builder.examples)} examples")
    print(f"Optimized agent: {len(optimized_agent.planner.prompt_builder.examples)} examples")
    print()
    
    test_questions = [
        "What is 75% of 120?",
        "Calculate 256 / 16",
        "What is 33 + 67?",
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 40)
        
        # Initial
        initial_agent.memory.clear()
        initial_result = await initial_agent.arun(question)
        print(f"Initial:   {initial_result}")
        
        # Optimized
        optimized_agent.memory.clear()
        optimized_result = await optimized_agent.arun(question)
        print(f"Optimized: {optimized_result}")
        print()


# =============================================================================
# Example 6: MIPRO Optimization
# =============================================================================

async def example_mipro_optimization():
    """
    Use MIPROv2 to optimize both instructions AND examples.
    
    MIPRO can automatically generate better role definitions/instructions
    in addition to selecting good examples.
    
    NOTE: Requires a DSPy-compatible LM for instruction generation.
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
    )
    from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
    from fairlib.utils.config_manager import save_agent_config
    
    from fair_prompt_optimizer import (
        AgentOptimizer,
        load_training_examples,
        numeric_accuracy,
    )
    
    import dspy
    
    print("=" * 60)
    print("EXAMPLE 6: MIPRO OPTIMIZATION")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Create agent
    # -------------------------------------------------------------------------
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    
    prompt_builder = PromptBuilder()
    prompt_builder.role_definition = RoleDefinition(
        "You are a math assistant that uses tools."
    )
    prompt_builder.format_instructions = [
        FormatInstruction(
            "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"...\", \"tool_input\": \"...\"}}"
        ),
    ]
    prompt_builder.examples = []
    
    planner = ReActPlanner(llm, tool_registry, prompt_builder=prompt_builder)
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=10,
    )
    
    save_agent_config(agent, "agent_for_mipro.json")
    print(f"Created agent for MIPRO optimization")
    print()
    
    # -------------------------------------------------------------------------
    # Configure DSPy LM for MIPRO
    # -------------------------------------------------------------------------
    # MIPRO needs a DSPy LM to generate instruction candidates
    # You can use OpenAI, Ollama, or other DSPy-compatible LMs
    
    try:
        # Example: Use Ollama with a local model
        dspy_lm = dspy.LM('ollama_chat/dolphin3-qwen25-3b', api_base='http://localhost:11434')
        print(f"Using DSPy LM: {dspy_lm}")
    except Exception as e:
        print(f"Could not configure DSPy LM: {e}")
        print("MIPRO requires a DSPy-compatible LM for instruction generation.")
        print("Options:")
        print("  - dspy.LM('ollama_chat/model', api_base='http://localhost:11434')")
        print("  - dspy.LM('openai/gpt-4')")
        return None
    
    # -------------------------------------------------------------------------
    # Run MIPRO optimization
    # -------------------------------------------------------------------------
    training_examples = load_training_examples(str(TRAINING_DATA_PATH))
    
    optimizer = AgentOptimizer(agent)
    
    print("Running MIPROv2 optimization (this may take a while)...")
    print()
    
    result = optimizer.compile(
        training_examples=training_examples[:5],
        metric=numeric_accuracy,
        optimizer="mipro",  # Use MIPRO instead of bootstrap
        mipro_auto="light",  # light, medium, or heavy
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        dspy_lm=dspy_lm,
    )
    
    result.save("agent_mipro_optimized.json")
    
    print()
    print(f"MIPRO optimization complete")
    print(f"Optimized examples: {len(result.examples)}")
    print(f"Role definition may have been updated")
    print(f"Saved to: agent_mipro_optimized.json")
    
    return optimizer


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
    
    # Example 1: Optimize a single agent (run this first!)
    print("Running: example_optimize_agent")
    asyncio.run(example_optimize_agent())
    
    # Example 2: Optimize a simple LLM (no agent)
    # print("Running: example_optimize_simple_llm")
    # asyncio.run(example_optimize_simple_llm())
    
    # Example 3: Optimize a multi-agent system
    # print("Running: example_optimize_multi_agent")
    # asyncio.run(example_optimize_multi_agent())
    
    # Example 4: Iterative optimization (requires example 1 first)
    # print("Running: example_iterative_optimization")
    # asyncio.run(example_iterative_optimization())
    
    # Example 5: Compare optimized vs non-optimized (requires example 1 first)
    # print("Running: example_compare_agents")
    # asyncio.run(example_compare_agents())
    
    # Example 6: MIPRO optimization (requires Ollama or OpenAI)
    # print("Running: example_mipro_optimization")
    # asyncio.run(example_mipro_optimization())