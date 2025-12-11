# fair_agent_module.py

"""
DSPy-compatible module that wraps fair_llm agents for prompt optimization
"""
import asyncio
import logging
from typing import Callable, List, Optional, Literal

import dspy

from .translator import (
    FAIRConfig,
    ModelConfig,
    AgentConfig,
    OptimizationMetadata,
    save_fair_config
)

logger = logging.getLogger(__name__)


class FAIRAgentModule(dspy.Module):
    """
    A DSPy Module that wraps a FAIR-LLM agent for optimization.
    
    This allows DSPy optimizers to optimize prompts while the actual
    execution uses the full FAIR-LLM pipeline with tools, RAG, workers, etc.
    """
    
    def __init__(self, agent):
        """
        Initialize the FAIR agent module.
        
        Args:
            agent: A FAIR-LLM agent instance (for now SimpleAgent, will need more here if we add different agent types)
        """
        super().__init__()
        
        self.agent = agent
        
        # Extract current role definition from agent
        self._extract_initial_config()
        
        # Create the DSPy signature for optimization
        self._create_signature()
        
        # Create the internal predictor (for DSPy compatibility)
        self.predict = dspy.Predict(self.signature)
    
    def _extract_initial_config(self):
        """Extract the current prompt and agent configuration."""
        # Extract role definition
        try:
            role_def = self.agent.planner.prompt_builder.role_definition
            self._initial_role = role_def.text
        except AttributeError:
            self._initial_role = "Complete the given task."
            logger.warning("Could not extract role_definition from agent, using default")
        
        # Extract model info
        try:
            llm = self.agent.llm
            self._model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            self._adapter_type = type(llm).__name__
        except AttributeError:
            self._model_name = "unknown"
            self._adapter_type = "unknown"
            logger.warning("Could not extract model_name or adapter_type from agent, using unknown")
        
        # Extract agent info
        try:
            self._agent_type = type(self.agent).__name__
            self._planner_type = type(self.agent.planner).__name__
            self._max_steps = getattr(self.agent, 'max_steps', 10)
        except AttributeError:
            self._agent_type = "SimpleAgent"
            self._planner_type = "SimpleReActPlanner"
            self._max_steps = 10
            logger.warning("Could not extract agent_type, planner_type or max_steps from agent, using default")
        
        # Extract tools
        try:
            tool_registry = self.agent.planner.tool_registry
            tools = tool_registry.get_all_tools()
            self._tools = [type(tool).__name__ for tool in tools.values()]
        except AttributeError:
            self._tools = []
            logger.warning("Could not extract tools from agent, tools are empty")
    
    def _create_signature(self):
        """Create a DSPy signature from the agent's current config."""
        signature_dict = {
            "__doc__": self._initial_role,
            "__annotations__": {
                "user_input": str,
                "response": str,
            },
            "user_input": dspy.InputField(desc="User's request or query"),
            "response": dspy.OutputField(desc="Agent's response"),
        }
        
        self.signature = type("FAIRAgentSignature", (dspy.Signature,), signature_dict)
    
    def _reset_agent_memory(self):
        """Reset the agent's memory for a fresh run."""
        try:
            self.agent.memory.clear()
        except Exception as e:
            logger.debug(f"Could not reset agent memory: {e}")
    
    def forward(self, user_input: str) -> dspy.Prediction:
        """
        Run the FAIR-LLM agent on the input.
        
        This is called by DSPy optimizers during training and evaluation. 
            [dspy.compile() calls this since this module inherits from dspy.Module]
        """
        # Reset agent memory for clean state
        self._reset_agent_memory()
        
        # Run agent
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.agent.arun(user_input))
                response = future.result()
        except RuntimeError:
            response = asyncio.run(self.agent.arun(user_input))
        
        # Extract result
        if isinstance(response, str):
            result = response
        elif hasattr(response, 'text'):
            result = response.text
        elif hasattr(response, 'content'):
            result = response.content
        elif hasattr(response, 'final_answer'):
            result = response.final_answer
        else:
            result = str(response)
        
        return dspy.Prediction(response=result)
    
    def get_config(self) -> FAIRConfig:
        """
        Extract the current prompt configuration.
        
        Returns a complete FAIRConfig that can recreate an agent.
        """
        role_definition = self._initial_role
        examples = []
        
        # Extract optimized instructions if available
        if hasattr(self, 'predict') and hasattr(self.predict, 'signature'):
            if hasattr(self.predict.signature, 'instructions') and self.predict.signature.instructions:
                role_definition = self.predict.signature.instructions
        
        # Extract optimized demos if available
        if hasattr(self, 'predict') and hasattr(self.predict, 'demos') and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, '_store') else demo
                parts = []
                if 'user_input' in demo_dict:
                    parts.append(f"User: {demo_dict['user_input']}")
                if 'reasoning' in demo_dict:
                    parts.append(f"Thought: {demo_dict['reasoning']}")
                if 'response' in demo_dict:
                    parts.append(f"Response: {demo_dict['response']}")
                if parts:
                    examples.append('\n'.join(parts))
        
        # Build complete config
        config = FAIRConfig(
            role_definition=role_definition,
            examples=examples,
            model=ModelConfig(
                model_name=self._model_name,
                adapter=self._adapter_type,
            ),
            agent=AgentConfig(
                agent_type=self._agent_type,
                planner_type=self._planner_type,
                max_steps=self._max_steps,
                tools=self._tools,
            ),
        )
        
        return config


class FAIRPromptOptimizer:
    """
    Prompt optimizer for FAIR-LLM agents.
    
    Wraps DSPy optimizers (BootstrapFewShot, MIPROv2) to optimize prompts
    while running the full FAIR-LLM agent execution pipeline.
    """
    
    def __init__(self, agent):
        """
        Initialize the prompt optimizer.
        
        Args:
            agent: A FAIR-LLM agent instance.
        """
        self.agent = agent
        self._module = None
    
    def compile(
        self,
        training_examples: List,
        metric: Callable,
        optimizer: str = "bootstrap",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        mipro_auto: Literal['light', 'medium', 'heavy'] | None = 'light',
        output_path: Optional[str] = None,
        dspy_lm = None # required to run mirpo optimization
    ) -> FAIRConfig:
        """
        Optimize the agent's prompts using the specified optimizer.
        
        Args:
            training_examples: List of TrainingExample or dspy.Example objects
            metric: Evaluation function (example, prediction, trace) -> bool|float
            optimizer: "bootstrap" for BootstrapFewShot, "mipro" for MIPROv2
            max_bootstrapped_demos: Max demos to generate from successful traces
            max_labeled_demos: Max labeled demos to include
            mipro_auto: MIPROv2 mode - "light", "medium", or "heavy"
            output_path: Optional path to save the optimized config
            dspy_lm: Model to use for mipro optimization
            
        Returns:
            FAIRConfig with optimized role_definition, examples, and full agent config
        """
        from dspy.teleprompt import BootstrapFewShot, MIPROv2
        from .translator import TrainingExample, DSPyTranslator
        from datetime import datetime
        
        # Create DSPy module wrapping our agent
        module = FAIRAgentModule(self.agent)
        
        # Convert training examples to DSPy format if needed
        translator = DSPyTranslator()
        if training_examples and isinstance(training_examples[0], TrainingExample):
            dspy_examples = translator.training_examples_to_dspy(training_examples)
        else:
            dspy_examples = training_examples
        
        # Run the appropriate optimizer
        if optimizer == "bootstrap":
            logger.info("Running BootstrapFewShot optimization\n")
            dspy_optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            optimized_module = dspy_optimizer.compile(module, trainset=dspy_examples)
            
        elif optimizer == "mipro":
            if dspy_lm is None:
                raise ValueError(
                    "MIPROv2 requires a DSPy LM for instruction generation. Exiting...\n"
                )

            logger.info(f"Running MIPROv2 ({mipro_auto}) optimization\n")
    
            dspy.configure(lm=dspy_lm)
            logger.info(f"Configured DSPy LM: {dspy_lm}")

            dspy_optimizer = MIPROv2(metric=metric, auto=mipro_auto)
            optimized_module = dspy_optimizer.compile(
                module,
                trainset=dspy_examples,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                requires_permission_to_run=False,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'bootstrap' or 'mipro'")
        
        # Store the optimized module
        self._module = optimized_module
        
        # Extract optimized config
        optimized_config = optimized_module.get_config()
        
        # Add optimization metadata
        optimized_config.metadata = OptimizationMetadata(
            optimized=True,
            optimized_at=datetime.now().isoformat(),
            optimizer=optimizer,
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
            },
            num_training_examples=len(training_examples),
        )
        
        # Save if path provided
        if output_path:
            save_fair_config(optimized_config, output_path)
            logger.info(f"Saved optimized config to {output_path}")
        
        return optimized_config