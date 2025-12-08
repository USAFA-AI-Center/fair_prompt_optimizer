# optimize_fair_agent.py

import asyncio

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

from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
from fair_prompt_optimizer.translator import TrainingExample
from fair_prompt_optimizer.metrics import numeric_accuracy


def main():
    print("=" * 60)
    print("FAIR-LLM Agent Prompt Optimization")
    print("=" * 60)
    
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
    
    optimized_config = optimizer.compile(
        training_examples=training_examples,
        metric=numeric_accuracy,
        optimizer="bootstrap",  # alternative: "mipro"
        max_bootstrapped_demos=4,
        output_path="prompts/math_agent_optimized.json"
    )
    
    # --- Step 4: Report results ---
    print(f"\nOptimization complete, saved to: prompts/math_agent_optimized.json\n")
    
    # --- Step 5: Apply optimized config back to agent ---
    print("\nApplying optimized config to agent...")
    optimizer.apply_to_agent(optimized_config)
    
    return agent, optimized_config


def test_agent(agent):
    print("\n" + "=" * 60)
    print("Testing Optimized Agent")
    print("=" * 60)
    print("Type 'exit' to quit.\n")
    
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
    agent, config = main()
    
    # Test the optimized agent interactively
    # test_agent(agent)