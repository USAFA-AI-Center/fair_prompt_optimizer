# optimize_fair_agent_mipro.py
"""
Example: Optimizing a FAIR-LLM agent's prompts using DSPy.

This demonstrates the full optimization flow built programatically:
1. Build the base FAIR agent (identical to all demos in fair_llm)
2. Pass it to FAIRPromptOptimizer
3. Run optimization with training examples
4. Save optimized config to JSON

You can then load the optimized agent from the generated config
"""

import asyncio
import dspy

# fair_llm imports
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition
)

from fair_prompt_optimizer import (
    FAIRPromptOptimizer,
    TrainingExample,
    numeric_accuracy,
    load_optimized_agent,
)

def main():
    # --- Step 1: Build the FAIR-LLM agent ---
    print("\nBuilding the FAIR-LLM agent...")
    
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are an expert mathematical calculator. Your job it is to perform mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. If a user's request requires "
        "multiple steps or tools, you must break it down and execute them sequentially. You must follow the strict formatting rules that follow..."
    )
    
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10
    )

    # --- Step 2: Create training examples ---
    print("\nCreating training examples...")
    
    training_examples = [
        TrainingExample(inputs={"user_input": "What is 15 + 27?"}, expected_output="42"),
        TrainingExample(inputs={"user_input": "Calculate 100 divided by 4"}, expected_output="25"),
        TrainingExample(inputs={"user_input": "What is 8 times 7?"}, expected_output="56"),
        TrainingExample(inputs={"user_input": "What is 144 minus 89?"}, expected_output="55"),
        TrainingExample(inputs={"user_input": "What is 25% of 200?"}, expected_output="50"),
        TrainingExample(inputs={"user_input": "Calculate the square root of 81"}, expected_output="9"),
        TrainingExample(inputs={"user_input": "What is 2 to the power of 8?"}, expected_output="256"),
        TrainingExample(inputs={"user_input": "What is 1000 - 347?"}, expected_output="653"),
    ]
    
    # --- Step 3: Create optimizer and run ---
    print("\nCreating FAIRPromptOptimizer...")
    
    optimizer = FAIRPromptOptimizer(agent)
    
    print("\nRunning optimization...")
    
    dspy_lm = dspy.LM("ollama/dolphin-llama3:8b", api_base="http://localhost:11434")
    optimized_config = optimizer.compile(
        training_examples=training_examples,
        metric=numeric_accuracy,
        optimizer="mipro",
        mipro_auto="light",
        max_bootstrapped_demos=4,
        output_path="prompts/math_agent_optimized.json",
        dspy_lm=dspy_lm,
    )
    
    # --- Step 4: Report results ---
    print(f"\nOptimization complete!")
    print(f"   Saved to: prompts/math_agent_optimized.json")
    print(f"   Role definition: {optimized_config.role_definition[:80]}...")
    print(f"   Examples: {len(optimized_config.examples)} demos")
    print(f"   Model: {optimized_config.model.adapter}({optimized_config.model.model_name})")
    print(f"   Tools: {optimized_config.agent.tools}")
    
    return optimized_config


def test_optimized_agent():
    """
    Load an optimized agent from config and test interactively.
    
    This demonstrates how to use load_optimized_agent() to recreate
    a fully configured agent from a saved config file.
    """
    print("Testing Optimized Agent.\nType 'exit' to quit.\n")
    
    # Load agent from saved config
    agent = load_optimized_agent("prompts/math_agent_optimized.json")
    
    async def run_loop():
        while True:
            try:
                user_input = input("👤 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("🤖 Agent: Goodbye!")
                    break
                
                response = await agent.arun(user_input)
                print(f"🤖 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Agent: Exiting...")
                break
    
    asyncio.run(run_loop())


if __name__ == "__main__":
    # Run optimization
    config = main()
    # test with the generated config
    # test_optimized_agent()