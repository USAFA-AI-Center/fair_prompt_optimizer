# fair_prompt_optimizer/optimizers/agent.py
"""
Agent optimizer and DSPy module.

This module provides:
- AgentModule: DSPy module wrapping a fairlib SimpleAgent
- AgentOptimizer: Optimizer for fairlib SimpleAgent with tools

Use this when you have an agent with tools and want to optimize
its prompts while running the full agent pipeline.
"""

import logging
from typing import Callable, Dict, List, Literal, Optional

import dspy
from fairlib.core.prompts import PromptBuilder
from fairlib.modules.agent.simple_agent import SimpleAgent

from ..config import (
    DSPyTranslator,
    OptimizedConfig,
    OptimizedPrompts,
    TrainingExample,
    compute_file_hash,
)
from .base import (
    clear_cuda_memory,
    combine_prompt_components,
    parse_optimized_prompt,
    run_async,
)

logger = logging.getLogger(__name__)


class AgentModule(dspy.Module):
    """
    DSPy Module wrapping a fairlib SimpleAgent.

    Exposes only the FinalAnswer to DSPy. The ReAct loop
    (Thought → Action → Observation cycles) is internal.
    """

    def __init__(
        self, agent: "SimpleAgent", input_field: str = "user_input", output_field: str = "response"
    ):
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
            self._initial_role = role_def.text if hasattr(role_def, "text") else str(role_def)
        except AttributeError:
            self._initial_role = "Complete the given task."
            logger.warning("Could not extract role_definition from agent, using default")

        # Extract format instructions
        try:
            format_instructions = self.agent.planner.prompt_builder.format_instructions
            self._initial_format_instructions = []
            for fi in format_instructions:
                if hasattr(fi, "text"):
                    self._initial_format_instructions.append(fi.text)
                elif hasattr(fi, "content"):
                    self._initial_format_instructions.append(fi.content)
                else:
                    self._initial_format_instructions.append(str(fi))
        except AttributeError:
            self._initial_format_instructions = []
            logger.debug("Could not extract format_instructions from agent")

        # Extract model info
        try:
            llm = self.agent.llm
            self._model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
            self._adapter_type = type(llm).__name__
        except AttributeError:
            self._model_name = "unknown"
            self._adapter_type = "unknown"
            logger.warning("Could not extract model info from agent")

        # Extract agent info
        try:
            self._agent_type = type(self.agent).__name__
            self._planner_type = type(self.agent.planner).__name__
            self._max_steps = getattr(self.agent, "max_steps", 10)
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
        # Combine all prompt components for optimization
        if self._initial_format_instructions:
            combined_instructions = combine_prompt_components(
                self._initial_role,
                self._initial_format_instructions,
            )
        else:
            combined_instructions = self._initial_role

        signature_dict = {
            "__doc__": combined_instructions,
            "__annotations__": {
                self.input_field: str,
                self.output_field: str,
            },
            self.input_field: dspy.InputField(desc="User's request or query"),
            self.output_field: dspy.OutputField(desc="Agent's response"),
        }
        self.signature = type("AgentSignature", (dspy.Signature,), signature_dict)

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

    def get_prompt_builder(self) -> Optional["PromptBuilder"]:
        """Get the agent's PromptBuilder."""
        try:
            return self.agent.planner.prompt_builder
        except AttributeError:
            return None

    def get_optimized_role(self) -> str:
        """Get the optimized role definition (instructions) if available.

        Note: For full prompt component access, use get_optimized_prompts() instead.
        """
        optimized = self.get_optimized_prompts()
        return optimized.role_definition or self._initial_role

    def get_optimized_prompts(self) -> OptimizedPrompts:
        """
        Get all optimized prompt components.

        Parses the optimized instructions back into individual components
        (role_definition, format_instructions) using section markers.
        """
        optimized_text = None
        if hasattr(self.predict, "signature") and hasattr(self.predict.signature, "instructions"):
            if self.predict.signature.instructions:
                optimized_text = self.predict.signature.instructions

        if optimized_text is None:
            return OptimizedPrompts(
                role_definition=self._initial_role,
                format_instructions=self._initial_format_instructions,
                role_definition_changed=False,
                format_instructions_changed=False,
            )

        return parse_optimized_prompt(
            optimized_text,
            self._initial_role,
            self._initial_format_instructions,
        )

    def _reset_agent_memory(self):
        """Reset the agent's memory for a fresh run."""
        try:
            self.agent.memory.clear()
        except Exception as e:
            logger.debug(f"Could not reset agent memory: {e}")

    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, "demos") and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, "_store") else demo
                demos.append(demo_dict)
        return demos


class AgentOptimizer:
    """
    Optimizer for fairlib SimpleAgent.

    Optimizes the agent's prompts (role_definition, examples) while
    running the full agent pipeline with tools.
    """

    def __init__(self, agent: "SimpleAgent", config: Optional[OptimizedConfig] = None):
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
    def from_config_file(cls, path: str, llm) -> "AgentOptimizer":
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
        mipro_auto: Literal["light", "medium", "heavy"] = "light",
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
        - Use DSPy MIPROv2 to optimize role_definition/instructions
        - Uses manual bootstrap for examples (No DSPy)

        For fairlib agents, training data should reflect a run through the fair_llm pipe.

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
        old_format_instructions = self.config.prompts.get("format_instructions", [])

        # Initialize optimized prompts with current values
        optimized_prompts = OptimizedPrompts(
            role_definition=old_role,
            format_instructions=old_format_instructions,
            role_definition_changed=False,
            format_instructions_changed=False,
        )

        logger.info(
            f"Running bootstrap to select full_trace examples ({len(training_examples)} candidates)"
        )
        # Run Bootstrap optimization
        examples = self._run_bootstrap(
            training_examples,
            metric,
            max_demos=max_bootstrapped_demos,
        )

        # Run MIPRO optimization
        if optimizer == "mipro":
            try:
                logger.info(f"Running MIPROv2 for instruction optimization (auto={mipro_auto})")
                optimized_prompts = self._optimize_instructions_mipro(
                    training_examples, metric, dspy_lm, mipro_auto, max_labeled_demos
                )
            except Exception:
                logger.error("Encountered error in MIRPOv2 optimization, using old role definition")

        # Update config
        if optimized_prompts.role_definition_changed and optimized_prompts.role_definition:
            self.config.role_definition = optimized_prompts.role_definition

        if optimized_prompts.format_instructions_changed and optimized_prompts.format_instructions:
            self.config.config["prompts"][
                "format_instructions"
            ] = optimized_prompts.format_instructions

        if examples:
            current_examples = list(self.config.examples)
            current_examples.extend(examples)
            self.config.examples = current_examples

        # Record provenance
        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, "__name__") else str(metric),
            training_data_path=training_data_path,
            training_data_hash=(
                compute_file_hash(training_data_path) if training_data_path else None
            ),
            num_examples=len(training_examples),
            examples_before=examples_before,
            examples_after=len(self.config.examples),
            role_definition_changed=optimized_prompts.role_definition_changed,
            format_instructions_changed=optimized_prompts.format_instructions_changed,
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
            },
        )

        logger.info(
            f"Optimization complete. Examples: {examples_before} → {len(self.config.examples)}"
        )
        if optimizer == "mipro":
            logger.info(f"Role definition changed: {optimized_prompts.role_definition_changed}")
            logger.info(
                f"Format instructions changed: {optimized_prompts.format_instructions_changed}"
            )

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

            user_input = ex.inputs.get("user_input", "")

            try:
                # Run the agent
                logger.debug(f"Running agent on: {user_input[:50]}...")
                prediction = module(user_input=user_input)

                # Create DSPy example for metric evaluation
                dspy_example = dspy.Example(
                    user_input=user_input,
                    expected_output=ex.expected_output,
                ).with_inputs("user_input")

                # Check metric
                score = metric(dspy_example, prediction, None)

                if score:
                    if ex.full_trace:
                        selected.append(ex.full_trace)
                        logger.info(f"Passed (full_trace): {user_input[:40]}...")
                    else:
                        logger.warning(
                            f"Passed but NO full_trace, training data must include a full trace example: {user_input[:40]}..."
                        )
                else:
                    logger.debug(f"Failed metric: {user_input[:40]}...")

            except Exception as e:
                logger.warning(f"Agent failed on '{user_input[:30]}...': {e}")

        logger.info(
            f"Bootstrap selected {len(selected)}/{min(len(training_examples), max_demos)} examples"
        )
        return selected

    def _optimize_instructions_mipro(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        dspy_lm,
        mipro_auto: str,
        max_labeled_demos: int,
    ) -> OptimizedPrompts:
        """
        Use MIPROv2 to optimize all prompt components (role_definition, format_instructions).

        The prompt components are combined into a structured format with XML-like markers,
        optimized by MIPRO, then parsed back into individual components.

        Note: MIPRO is for instruction optimization, not example selection.
        Examples always come from our manual bootstrap with full_trace.

        Returns:
            OptimizedPrompts containing all optimized components
        """
        import dspy
        from dspy.teleprompt import MIPROv2

        dspy.configure(lm=dspy_lm)

        module = AgentModule(self.agent)
        translator = DSPyTranslator(
            input_field=module.input_field, output_field=module.output_field
        )
        dspy_examples = translator.to_dspy_examples(training_examples)

        try:
            optimizer = MIPROv2(metric=metric, auto=mipro_auto)
            optimized = optimizer.compile(
                module,
                trainset=dspy_examples,
                max_labeled_demos=max_labeled_demos,
                requires_permission_to_run=False,
            )

            # Get all optimized prompt components
            optimized_prompts = optimized.get_optimized_prompts()

            logger.info("MIPRO optimization complete:")
            logger.info(f"  - role_definition changed: {optimized_prompts.role_definition_changed}")
            logger.info(
                f"  - format_instructions changed: {optimized_prompts.format_instructions_changed}"
            )

            return optimized_prompts

        except Exception as e:
            logger.warning(f"MIPRO instruction optimization failed: {e}")
            return OptimizedPrompts(
                role_definition=self.config.role_definition,
                format_instructions=self.config.prompts.get("format_instructions"),
                role_definition_changed=False,
                format_instructions_changed=False,
            )

    def test(self, user_input: str) -> str:
        """Run a test input through the agent."""
        module = self._module or AgentModule(self.agent)
        result = module(user_input=user_input)
        return result.response

    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)
