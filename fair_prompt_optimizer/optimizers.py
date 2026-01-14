# fair_prompt_optimizer/optimizers.py
# TODO:: design of user having to initialize an LLM from fair_llm is not great
"""
DSPy-compatible modules and optimizers for FAIR-LLM.

Three optimization levels:
1. SimpleLLMOptimizer - LLM with system prompt (no agent pipeline)
2. AgentOptimizer - Single SimpleAgent with tools
3. MultiAgentOptimizer - HierarchicalAgentRunner with manager + workers

All modules follow the FAIRAgentModule pattern:
- Create a DSPy signature from the component's config
- Create a dspy.Predict for DSPy to attach demos to
- forward() runs the real component and returns dspy.Prediction
- get_config() extracts optimized demos from self.predict.demos
"""

import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional

import dspy

from .config import (
    OptimizedConfig,
    TrainingExample,
    DSPyTranslator,
    compute_file_hash,
)

from fairlib.modules.agent.simple_agent import SimpleAgent
from fairlib.modules.agent.multi_agent_runner import HierarchicalAgentRunner 
from fairlib.core.prompts import PromptBuilder

logger = logging.getLogger(__name__)


def run_async(coro):
    """
    Run an async coroutine from sync code.
    
    Handles the case where we might already be in an async context.
    """
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(coro)

# Reset state for each given training example
def clear_cuda_memory():
    """Clear CUDA memory cache if torch is available."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class SimpleLLMModule(dspy.Module):
    """
    DSPy Module wrapping a simple LLM with system prompt.
    
    No agent pipeline, just direct LLM calls.
    """
    
    def __init__(self, llm, system_prompt: str, input_field: str = "user_input", output_field: str = "response"):
        """
        Initialize the SimpleLLM module.
        
        Args:
            llm: A fairlib LLM adapter (HuggingFaceAdapter, OpenAIAdapter, etc.)
            system_prompt: The system prompt to use
            input_field: Name of input field for DSPy signature
            output_field: Name of output field for DSPy signature
        """
        super().__init__()
        
        self.llm = llm
        self.system_prompt = system_prompt
        self.input_field = input_field
        self.output_field = output_field
        
        # Create DSPy signature
        self._create_signature()
        
        # Create predictor for DSPy to attach demos to
        self.predict = dspy.Predict(self.signature)
    
    def _create_signature(self):
        """Create a DSPy signature from the system prompt."""
        signature_dict = {
            "__doc__": self.system_prompt,
            "__annotations__": {
                self.input_field: str,
                self.output_field: str,
            },
            self.input_field: dspy.InputField(desc="User's request or query"),
            self.output_field: dspy.OutputField(desc="LLM's response"),
        }
        self.signature = type("SimpleLLMSignature", (dspy.Signature,), signature_dict)
    
    def forward(self, **kwargs) -> dspy.Prediction:
        """Run the LLM on the input."""
        from fairlib.core.message import Message
        
        user_input = kwargs.get(self.input_field, kwargs.get("user_input", ""))
        
        clear_cuda_memory()
        
        try:
            messages = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=user_input),
            ]
            response = self.llm.invoke(messages)
            
            # TODO:: debug here
            if hasattr(response, 'content'):
                result = response.content
            elif hasattr(response, 'text'):
                result = response.text
            else:
                result = str(response)
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            result = f"Error: {e}"
        
        return dspy.Prediction(**{self.output_field: result})
    
    def get_optimized_prompt(self) -> str:
        """Get the optimized system prompt (instructions) if available."""
        if hasattr(self.predict, 'signature') and hasattr(self.predict.signature, 'instructions'):
            if self.predict.signature.instructions:
                return self.predict.signature.instructions
        return self.system_prompt
    
    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, 'demos') and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, '_store') else demo
                demos.append(demo_dict)
        return demos


class AgentModule(dspy.Module):
    """
    DSPy Module wrapping a fairlib SimpleAgent.
    
    Exposes only the FinalAnswer to DSPy. The ReAct loop
    (Thought → Action → Observation cycles) is internal.
    """
    
    def __init__(self, agent: 'SimpleAgent', input_field: str = "user_input", output_field: str = "response"):
        """
        Initialize the Agent module.
        
        Args:
            agent: A fairlib SimpleAgent instance
            input_field: Name of input field for DSPy signature
            output_field: Name of output field for DSPy signature
        """
        super().__init__()
        
        self.agent = agent
        self.input_field = input_field
        self.output_field = output_field
        
        # Extract initial configuration
        self._extract_initial_config()
        
        # Create DSPy signature
        self._create_signature()
        
        # Create predictor for DSPy to attach demos to
        self.predict = dspy.Predict(self.signature)
    
    def _extract_initial_config(self):
        """Extract the current prompt and agent configuration."""
        # Extract role definition
        try:
            role_def = self.agent.planner.prompt_builder.role_definition
            self._initial_role = role_def.text if hasattr(role_def, 'text') else str(role_def)
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
            logger.warning("Could not extract model info from agent")
        
        # Extract agent info
        try:
            self._agent_type = type(self.agent).__name__
            self._planner_type = type(self.agent.planner).__name__
            self._max_steps = getattr(self.agent, 'max_steps', 10)
        except AttributeError:
            self._agent_type = "SimpleAgent"
            self._planner_type = "SimpleReActPlanner"
            self._max_steps = 10
        
        # Extract tools
        try:
            tool_registry = self.agent.planner.tool_registry
            tools = tool_registry.get_all_tools()
            self._tools = [type(tool).__name__ for tool in tools.values()]
        except AttributeError:
            self._tools = []
    
    def _create_signature(self):
        """Create a DSPy signature from the agent's current config."""
        signature_dict = {
            "__doc__": self._initial_role,
            "__annotations__": {
                self.input_field: str,
                self.output_field: str,
            },
            self.input_field: dspy.InputField(desc="User's request or query"),
            self.output_field: dspy.OutputField(desc="Agent's response"),
        }
        self.signature = type("AgentSignature", (dspy.Signature,), signature_dict)
    
    def _reset_agent_memory(self):
        """Reset the agent's memory for a fresh run."""
        try:
            self.agent.memory.clear()
        except Exception as e:
            logger.debug(f"Could not reset agent memory: {e}")
    
    def forward(self, **kwargs) -> dspy.Prediction:
        """Run the agent and return only FinalAnswer."""
        user_input = kwargs.get(self.input_field, kwargs.get("user_input", ""))
        
        # Reset agent memory for clean state
        self._reset_agent_memory()
        clear_cuda_memory()
        
        try:
            result = run_async(self.agent.arun(user_input))
        except Exception as e:
            logger.error(f"Agent error: {e}")
            result = f"Error: {e}"
        
        return dspy.Prediction(**{self.output_field: result})
    
    def get_prompt_builder(self) -> Optional['PromptBuilder']:
        """Get the agent's PromptBuilder."""
        try:
            return self.agent.planner.prompt_builder
        except AttributeError:
            return None
    
    def get_optimized_role(self) -> str:
        """Get the optimized role definition (instructions) if available."""
        if hasattr(self.predict, 'signature') and hasattr(self.predict.signature, 'instructions'):
            if self.predict.signature.instructions:
                return self.predict.signature.instructions
        return self._initial_role
    
    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, 'demos') and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, '_store') else demo
                demos.append(demo_dict)
        return demos


class MultiAgentModule(dspy.Module):
    """
    DSPy Module wrapping a fairlib HierarchicalAgentRunner.
    
    Exposes only the final result to DSPy. The multi-agent coordination
    (manager delegating to workers) is internal.

    #TODO:: make this recursively optimize from running agent modules [as a togggle mode, will take so long to run]
    """
    
    def __init__(self, runner: 'HierarchicalAgentRunner', input_field: str = "user_input", output_field: str = "response"):
        """
        Initialize the MultiAgent module.
        
        Args:
            runner: A fairlib HierarchicalAgentRunner instance
            input_field: Name of input field for DSPy signature
            output_field: Name of output field for DSPy signature
        """
        super().__init__()
        
        self.runner = runner
        self.input_field = input_field
        self.output_field = output_field
        
        # Extract initial configuration
        self._extract_initial_config()
        
        # Create DSPy signature
        self._create_signature()
        
        # Create predictor for DSPy to attach demos to
        self.predict = dspy.Predict(self.signature)
    
    def _extract_initial_config(self):
        """Extract the current configuration from the runner."""
        # Extract manager role
        try:
            manager = self.runner.manager
            role_def = manager.planner.prompt_builder.role_definition
            self._manager_role = role_def.text if hasattr(role_def, 'text') else str(role_def)
        except AttributeError:
            self._manager_role = "Coordinate workers to complete the task."
            logger.warning("Could not extract manager role, using default")
        
        # Extract worker info
        self._worker_roles = {}
        try:
            for name, worker in self.runner.workers.items():
                role_def = worker.planner.prompt_builder.role_definition
                self._worker_roles[name] = role_def.text if hasattr(role_def, 'text') else str(role_def)
        except AttributeError:
            logger.warning("Could not extract worker roles")
    
    def _create_signature(self):
        """Create a DSPy signature for the multi-agent system."""
        signature_dict = {
            "__doc__": self._manager_role,
            "__annotations__": {
                self.input_field: str,
                self.output_field: str,
            },
            self.input_field: dspy.InputField(desc="User's request or query"),
            self.output_field: dspy.OutputField(desc="Multi-agent system's response"),
        }
        self.signature = type("MultiAgentSignature", (dspy.Signature,), signature_dict)
    
    def _reset_runner_memory(self):
        """Reset memory for all agents in the runner."""
        try:
            self.runner.manager.memory.clear()
            for worker in self.runner.workers.values():
                worker.memory.clear()
        except Exception as e:
            logger.debug(f"Could not reset runner memory: {e}")
    
    def forward(self, **kwargs) -> dspy.Prediction:
        """Run the multi-agent system and return the final result."""
        user_input = kwargs.get(self.input_field, kwargs.get("user_input", ""))
        
        # Reset memory for clean state
        self._reset_runner_memory()
        clear_cuda_memory()
        
        try:
            result = run_async(self.runner.run(user_input))
        except Exception as e:
            logger.error(f"MultiAgent error: {e}")
            result = f"Error: {e}"
        
        # Extract string result
        if isinstance(result, str):
            response = result
        elif hasattr(result, 'text'):
            response = result.text
        elif hasattr(result, 'content'):
            response = result.content
        else:
            response = str(result)
        
        return dspy.Prediction(**{self.output_field: response})
    
    def get_manager_builder(self) -> Optional['PromptBuilder']:
        """Get the manager's PromptBuilder."""
        try:
            return self.runner.manager.planner.prompt_builder
        except AttributeError:
            return None
    
    def get_worker_builders(self) -> Dict[str, 'PromptBuilder']:
        """Get PromptBuilders for all workers."""
        builders = {}
        try:
            for name, worker in self.runner.workers.items():
                builders[name] = worker.planner.prompt_builder
        except AttributeError:
            pass
        return builders
    
    def get_optimized_manager_role(self) -> str:
        """Get the optimized manager role (instructions) if available."""
        if hasattr(self.predict, 'signature') and hasattr(self.predict.signature, 'instructions'):
            if self.predict.signature.instructions:
                return self.predict.signature.instructions
        return self._manager_role
    
    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, 'demos') and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, '_store') else demo
                demos.append(demo_dict)
        return demos


class SimpleLLMOptimizer:
    """
    Optimizer for simple LLM + system prompt.
    
    Use this when you just have an LLM with a system prompt and want
    to optimize the prompt with few-shot examples.
    
    Unlike AgentOptimizer, this does NOT require full_trace because there's
    no agent workflow to demonstrate - just simple input/output pairs.
    """
    
    def __init__(self, llm, system_prompt: str, config: Optional[OptimizedConfig] = None):
        """
        Initialize the SimpleLLM optimizer.
        
        Args:
            llm: A fairlib LLM adapter
            system_prompt: The system prompt to optimize
            config: Optional existing config to build upon
        """
        self.llm = llm
        self.system_prompt = system_prompt
        
        if config:
            self.config = config
        else:
            # Extract model info from LLM adapter
            model_info = self._extract_model_info(llm)
            
            self.config = OptimizedConfig(config={
                "version": "1.0", # TODO:: Persist this verison somewhere permanent
                "type": "simple_llm",
                "prompts": {
                    "system_prompt": system_prompt,
                    "examples": [],
                },
                "model": model_info,
            })
            
        self._module = None
    
    def _extract_model_info(self, llm) -> Dict[str, Any]:
        """Extract model information from LLM adapter."""
        adapter_name = type(llm).__name__
        
        # Try to get model name from common attributes
        model_name = getattr(llm, 'model_name', None) or \
                     getattr(llm, 'model_id', None) or \
                     getattr(llm, 'model', None) or \
                     "unknown"
        
        return {
            "adapter": adapter_name,
            "model_name": str(model_name),
            "adapter_kwargs": {},
        }
    
    def compile(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        optimizer: str = "bootstrap",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        mipro_auto: Literal['light', 'medium', 'heavy'] = 'light',
        training_data_path: Optional[str] = None,
        dspy_lm=None,
    ) -> OptimizedConfig:
        """
        Optimize the system prompt.
        
        Args:
            training_examples: Input/output examples
            metric: Evaluation function (example, prediction, trace) -> bool/float
            optimizer: "bootstrap" or "mipro"
            max_bootstrapped_demos: Max demos to generate
            max_labeled_demos: Max labeled demos
            mipro_auto: MIPROv2 intensity
            training_data_path: Path to training data (for provenance)
            dspy_lm: DSPy LM for MIPROv2
            
        Returns:
            OptimizedConfig with updated prompts and provenance
        """
        from dspy.teleprompt import BootstrapFewShot, MIPROv2
        
        # Early validation
        if optimizer not in ("bootstrap", "mipro"):
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'bootstrap' or 'mipro'")
        
        if optimizer == "mipro" and dspy_lm is None:
            raise ValueError("MIPROv2 requires dspy_lm parameter")
        
        # Create DSPy module
        module = SimpleLLMModule(self.llm, self.system_prompt)
        
        # Convert to DSPy format
        translator = DSPyTranslator()
        dspy_examples = translator.to_dspy_examples(training_examples)
        
        # Track starting state
        examples_before = len(self.config.prompts.get("examples", []))
        old_prompt = self.system_prompt
        
        logger.info(f"Starting SimpleLLM optimization with {len(dspy_examples)} examples")
        
        # Run optimization
        if optimizer == "bootstrap":
            dspy_optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            optimized_module = dspy_optimizer.compile(module, trainset=dspy_examples)
            
        elif optimizer == "mipro":
            dspy.configure(lm=dspy_lm)
            dspy_optimizer = MIPROv2(metric=metric, auto=mipro_auto)
            optimized_module = dspy_optimizer.compile(
                module,
                trainset=dspy_examples,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                requires_permission_to_run=False,
            )
        
        self._module = optimized_module
        
        # Extract optimized config
        new_prompt = optimized_module.get_optimized_prompt()
        # TODO:: convert these demos to fairlib formatted strings before inserting to config
        demos = optimized_module.get_demos()
        
        # Update config
        self.config.config["prompts"]["system_prompt"] = new_prompt
        self.config.config["prompts"]["examples"] = demos 
        
        # Record provenance
        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, '__name__') else str(metric),
            training_data_path=training_data_path,
            training_data_hash=compute_file_hash(training_data_path) if training_data_path else None,
            num_examples=len(training_examples),
            examples_before=examples_before,
            examples_after=len(demos),
            role_definition_changed=(new_prompt != old_prompt),
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
            },
        )
        
        logger.info(f"Optimization complete. Examples: {examples_before} → {len(demos)}")
        
        return self.config
    
    def test(self, user_input: str) -> str:
        """Run a test input through the LLM."""
        module = self._module or SimpleLLMModule(self.llm, self.system_prompt)
        result = module(user_input=user_input)
        return result.response
    
    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)


class AgentOptimizer:
    """
    Optimizer for fairlib SimpleAgent.
    
    Optimizes the agent's prompts (role_definition, examples) while
    running the full agent pipeline with tools.
    """
    
    def __init__(self, agent: 'SimpleAgent', config: Optional[OptimizedConfig] = None):
        """
        Initialize the Agent optimizer.
        
        Args:
            agent: A fairlib SimpleAgent instance
            config: Optional existing config to build upon
        """
        self.agent = agent
        self.config = config or OptimizedConfig.from_agent(agent)
        self._module = None
    
    @classmethod
    def from_config_file(cls, path: str, llm) -> 'AgentOptimizer':
        """Create optimizer by loading config file."""
        from fairlib.utils.config_manager import load_agent
        
        agent = load_agent(path, llm)
        config = OptimizedConfig.from_file(path)
        return cls(agent, config)
    
    def compile(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        optimizer: str = "bootstrap",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        mipro_auto: Literal['light', 'medium', 'heavy'] = 'light',
        training_data_path: Optional[str] = None,
        dspy_lm=None,
    ) -> OptimizedConfig:
        """
        Optimize the agent's prompts.
        
        Two optimization modes:
        
        Bootstrap (optimizer="bootstrap"):
        - Runs agent on each training example
        - If output passes metric, injects the pre-written full_trace
        - Requires full_trace in training examples
        - This mode does not leverage DSPy
        
        MIPRO (optimizer="mipro"):
        - Use DSPy MIPROv2 to optimize role_definition/instructions # TODO:: need to extend to all parts of prompt from PromptBuilder
        - Uses manual bootstrap for examples (No DSPy)
        
        For fairlib agents, training data should reflect a run through the fair_llm pipe. # TODO:: could this be serialized in some way?
        
        Args:
            training_examples: Examples with full_trace showing complete agent workflow
            metric: Evaluation function (example, prediction, trace) -> bool/float
            optimizer: "bootstrap" or "mipro"
            max_bootstrapped_demos: Max demos to include
            max_labeled_demos: Max labeled demos (for MIPRO instruction candidates)
            mipro_auto: MIPROv2 intensity ('light', 'medium', 'heavy')
            training_data_path: Path to training data (for provenance)
            dspy_lm: DSPy LM for MIPROv2 instruction generation
            
        Returns:
            OptimizedConfig with optimized prompts
        """
        if optimizer not in ("bootstrap", "mipro"):
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'bootstrap' or 'mipro'")
        
        if optimizer == "mipro" and dspy_lm is None:
            raise ValueError("MIPROv2 requires dspy_lm parameter")
        
        # Full conversation trace mandatory
        has_full_trace = any(ex.full_trace for ex in training_examples)
        if not has_full_trace:
            logger.warning(
                "Agent optimization requires full_trace in training examples. "
                "Without full_trace, the model won't learn proper tool usage patterns."
            )
        
        # Track starting state
        examples_before = len(self.config.examples)
        old_role = self.config.role_definition
        new_role = old_role
        
        logger.info(f"Running bootstrap to select full_trace examples ({len(training_examples)} candidates)")
        # TODO:: current pattern does not allow user to have MIPRO without bootstrapping, add option to reduce run time in use cases
        # Run Bootstrap optimization
        examples = self._run_bootstrap(
            training_examples,
            metric,
            max_demos=max_bootstrapped_demos,
        )
        
        # Run MIPRO optimization 
        if optimizer == "mipro":
            logger.info(f"Running MIPROv2 for instruction optimization (auto={mipro_auto})")
            new_role = self._optimize_instructions_mipro(
                training_examples, metric, dspy_lm, mipro_auto, max_labeled_demos
            )
        
        # Update config
        if new_role != old_role:
            self.config.role_definition = new_role
        
        if examples:
            current_examples = list(self.config.examples)
            current_examples.extend(examples)
            self.config.examples = current_examples
        
        # Record provenance
        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, '__name__') else str(metric),
            training_data_path=training_data_path,
            training_data_hash=compute_file_hash(training_data_path) if training_data_path else None,
            num_examples=len(training_examples),
            examples_before=examples_before,
            examples_after=len(self.config.examples),
            role_definition_changed=(new_role != old_role),
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
            },
        )
        
        logger.info(f"Optimization complete. Examples: {examples_before} → {len(self.config.examples)}")
        if optimizer == "mipro":
            logger.info(f"Role definition changed: {new_role != old_role}")
        
        return self.config
    
    def _run_bootstrap(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        max_demos: int,
    ) -> List[str]:
        """
        Manual bootstrap loop: run agent, check metric, inject full_trace.
        
        This is our implementation of the bootstrap algorithm that allows us
        to use pre-written full_trace examples instead of DSPy's captured traces.
        """
        import dspy
        
        selected = []
        module = AgentModule(self.agent)
        
        for ex in training_examples:
            if len(selected) >= max_demos:
                break
            
            user_input = ex.inputs.get('user_input', '')
            
            try:
                # Run the agent
                logger.debug(f"Running agent on: {user_input[:50]}...")
                prediction = module(user_input=user_input)
                
                # Create DSPy example for metric evaluation
                dspy_example = dspy.Example(
                    user_input=user_input,
                    expected_output=ex.expected_output,
                ).with_inputs('user_input')
                
                # Check metric
                score = metric(dspy_example, prediction, None)
                
                if score: # TODO:: this acceptance needs to be tunable, some metrics may output floats
                    if ex.full_trace:
                        selected.append(ex.full_trace)
                        logger.info(f"Passed (full_trace): {user_input[:40]}...")
                    else:
                        logger.warning(f"Passed but NO full_trace, training data must include a full trace example: {user_input[:40]}...")
                else:
                    logger.debug(f"Failed metric: {user_input[:40]}...")
                    
            except Exception as e:
                logger.warning(f"Agent failed on '{user_input[:30]}...': {e}")
        
        logger.info(f"Bootstrap selected {len(selected)}/{min(len(training_examples), max_demos)} examples")
        return selected
    
    def _optimize_instructions_mipro(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        dspy_lm,
        mipro_auto: str,
        max_labeled_demos: int,
    ) -> str:
        """
        Use MIPROv2 to optimize role_definition/instructions. #TODO:: extend this to all prompt parts
        
        Note: MIPRO is for instruction optimization, not example selection.
        Examples always come from our manual bootstrap with full_trace.
        """
        import dspy
        from dspy.teleprompt import MIPROv2
        
        dspy.configure(lm=dspy_lm)
        
        module = AgentModule(self.agent)
        translator = DSPyTranslator()
        dspy_examples = translator.to_dspy_examples(training_examples)
        
        try:
            optimizer = MIPROv2(metric=metric, auto=mipro_auto)
            optimized = optimizer.compile(
                module,
                trainset=dspy_examples,
                max_labeled_demos=max_labeled_demos,
                requires_permission_to_run=False,
            )
            new_role = optimized.get_optimized_role()
            logger.info(f"MIPRO optimized role definition")
            # TODO:: here we return all pieces of the prompt builder [tool_instructions, etc.]
            return new_role
        except Exception as e:
            logger.warning(f"MIPRO instruction optimization failed: {e}")
            return self.config.role_definition
    
    def test(self, user_input: str) -> str:
        """Run a test input through the agent."""
        module = self._module or AgentModule(self.agent)
        result = module(user_input=user_input)
        return result.response
    
    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)


class MultiAgentOptimizer:
    """
    Optimizer for fairlib HierarchicalAgentRunner.
    
    Optimizes the manager and/or worker prompts while running
    the full multi-agent coordination pipeline.

    #TODO:: this only optimizes at the manager level. We need to extend this to optmize over Agent
    #TODO:: optmizers as well. [Should be a configurable mode]
    """
    
    def __init__(
        self,
        runner: 'HierarchicalAgentRunner',
        config: Optional[OptimizedConfig] = None,
        optimize_manager: bool = True,
        optimize_workers: bool = False,
    ):
        """
        Initialize the MultiAgent optimizer.
        
        Args:
            runner: A fairlib HierarchicalAgentRunner instance
            config: Optional existing config to build upon
            optimize_manager: Whether to optimize the manager's prompts
            optimize_workers: Whether to optimize worker prompts
        """
        self.runner = runner
        self.optimize_manager = optimize_manager
        self.optimize_workers = optimize_workers
        
        # Extract config from runner
        if config:
            self.config = config
        else:
            from .config import extract_multi_agent_config
            config_dict = extract_multi_agent_config(runner)
            self.config = OptimizedConfig(config=config_dict)
        
        self._module = None

    # TODO:: class method to load from config
    
    def compile(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        optimizer: str = "bootstrap",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        mipro_auto: Literal['light', 'medium', 'heavy'] = 'light',
        training_data_path: Optional[str] = None,
        dspy_lm=None,
        optimize_manager: Optional[bool] = None,
        optimize_workers: Optional[bool] = None,
    ) -> OptimizedConfig:
        """
        Optimize the multi-agent system's prompts.
        
        Two optimization modes:
        
        Bootstrap (optimizer="bootstrap"):
        - Runs agent on each training example
        - If output passes metric, injects the pre-written full_trace
        - Requires full_trace in training examples
        - This mode does not leverage DSPy
        
        MIPRO (optimizer="mipro"):
        - Use DSPy MIPROv2 to optimize role_definition/instructions # TODO:: need to extend to all parts of prompt from PromptBuilder
        - Uses manual bootstrap for examples (No DSPy)
        
        For fairlib agents, training data should reflect a run through the fair_llm pipe. # TODO:: could this be serialized in some way?
        
        Args:
            training_examples: Examples with full_trace showing complete agent workflow
            metric: Evaluation function (example, prediction, trace) -> bool/float
            optimizer: "bootstrap" or "mipro"
            max_bootstrapped_demos: Max demos to include
            max_labeled_demos: Max labeled demos (for MIPRO instruction candidates)
            mipro_auto: MIPROv2 intensity ('light', 'medium', 'heavy')
            training_data_path: Path to training data (for provenance)
            dspy_lm: DSPy LM for MIPROv2 instruction generation
            
        Returns:
            OptimizedConfig with optimized prompts
        """
        # Use instance defaults if not specified
        if optimize_manager is None:
            optimize_manager = self.optimize_manager
        if optimize_workers is None:
            optimize_workers = self.optimize_workers

        if optimizer not in ("bootstrap", "mipro"):
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'bootstrap' or 'mipro'")
        
        if optimizer == "mipro" and dspy_lm is None:
            raise ValueError("MIPROv2 requires dspy_lm parameter")
        
        # Check for full_trace (required for multi-agent)
        has_full_trace = any(ex.full_trace for ex in training_examples)
        if not has_full_trace:
            logger.warning(
                "Multi-agent optimization requires full_trace in training examples."
            )
        
        # Track starting state
        manager_prompts = self.config.config.get("manager", {}).get("prompts", {})
        examples_before = len(manager_prompts.get("examples", []))
        old_role = manager_prompts.get("role_definition", "")
        new_role = old_role
        
        # TODO:: current pattern does not allow user to have MIPRO without bootstrapping, add option to reduce run time in use cases
        # Run Bootstrap optimization
        logger.info(f"Running bootstrap to select full_trace examples ({len(training_examples)} candidates)")
        examples = self._run_bootstrap(
            training_examples,
            metric,
            max_demos=max_bootstrapped_demos,
        )
        
        # Run MIPRO optimization
        if optimizer == "mipro":
            logger.info(f"Running MIPROv2 for manager instruction optimization (auto={mipro_auto})")
            new_role = self._optimize_instructions_mipro(
                training_examples, metric, dspy_lm, mipro_auto, max_labeled_demos
            )
        
        # Update config TODO:: extend config updates to all agents recursively
        if optimize_manager:
            if "manager" not in self.config.config:
                self.config.config["manager"] = {"prompts": {}}
            if "prompts" not in self.config.config["manager"]:
                self.config.config["manager"]["prompts"] = {}
            
            if new_role != old_role:
                self.config.config["manager"]["prompts"]["role_definition"] = new_role
            
            if examples:
                current = self.config.config["manager"]["prompts"].get("examples", [])
                current.extend(examples)
                self.config.config["manager"]["prompts"]["examples"] = current
        
        # Record provenance
        examples_after = len(self.config.config.get("manager", {}).get("prompts", {}).get("examples", []))
        
        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, '__name__') else str(metric),
            training_data_path=training_data_path,
            training_data_hash=compute_file_hash(training_data_path) if training_data_path else None,
            num_examples=len(training_examples),
            examples_before=examples_before,
            examples_after=examples_after,
            role_definition_changed=(new_role != old_role),
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
                "optimize_manager": optimize_manager,
                "optimize_workers": optimize_workers,
            },
        )
        
        logger.info(f"Optimization complete. Examples: {examples_before} → {examples_after}")
        if optimizer == "mipro":
            logger.info(f"Manager role definition changed: {new_role != old_role}")
        
        return self.config
    
    def _run_bootstrap(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        max_demos: int,
    ) -> List[str]:
        """
        Manual bootstrap loop for multi-agent: run system, check metric, inject full_trace.
        """
        import dspy
        
        selected = []
        module = MultiAgentModule(self.runner)
        
        for ex in training_examples:
            if len(selected) >= max_demos:
                break
            
            user_input = ex.inputs.get('user_input', '')
            
            try:
                # Run the full multi-agent system
                logger.debug(f"Running multi-agent on: {user_input[:50]}...")
                prediction = module(user_input=user_input)
                
                # Check metric
                dspy_example = dspy.Example(
                    user_input=user_input,
                    response=ex.expected_output,
                ).with_inputs('user_input')
                
                score = metric(dspy_example, prediction, None)
                
                if score:
                    if ex.full_trace:
                        selected.append(ex.full_trace)
                        logger.info(f"Passed (full_trace): {user_input[:40]}...")
                    else:
                        logger.warning(f"Passed but NO full_trace: {user_input[:40]}...")
                else:
                    logger.debug(f"Failed metric: {user_input[:40]}...")
                    
            except Exception as e:
                logger.warning(f"Multi-agent failed on '{user_input[:30]}...': {e}")
        
        logger.info(f"Bootstrap selected {len(selected)}/{min(len(training_examples), max_demos)} examples")
        return selected
    
    def _optimize_instructions_mipro(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        dspy_lm,
        mipro_auto: str,
        max_labeled_demos: int,
    ) -> str:
        """
        Use MIPROv2 to optimize manager role_definition/instructions only.
        """
        import dspy
        from dspy.teleprompt import MIPROv2
        
        dspy.configure(lm=dspy_lm)
        
        module = MultiAgentModule(self.runner)
        translator = DSPyTranslator()
        dspy_examples = translator.to_dspy_examples(training_examples)
        
        try:
            optimizer = MIPROv2(metric=metric, auto=mipro_auto)
            optimized = optimizer.compile(
                module,
                trainset=dspy_examples,
                max_labeled_demos=max_labeled_demos,
                requires_permission_to_run=False,
            )
            new_role = optimized.get_optimized_manager_role()
            # TODO:: return all parts of the promptBuilder
            logger.info(f"MIPRO optimized manager role definition")
            return new_role
        except Exception as e:
            logger.warning(f"MIPRO instruction optimization failed: {e}")
            return self.config.config.get("manager", {}).get("prompts", {}).get("role_definition", "")
    
    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)