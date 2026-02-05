# fair_prompt_optimizer/optimizers/multi_agent.py
"""
Multi-agent optimizer and DSPy module.

This module provides:
- MultiAgentModule: DSPy module wrapping a fairlib HierarchicalAgentRunner
- MultiAgentOptimizer: Optimizer for multi-agent systems (manager + workers)

Use this when you have a multi-agent system and want to optimize
the manager's prompts while running the full coordination pipeline.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import dspy
from fairlib.core.prompts import PromptBuilder
from fairlib.modules.agent.multi_agent_runner import HierarchicalAgentRunner

from ..config import (
    DSPyTranslator,
    OptimizedConfig,
    TrainingExample,
    compute_file_hash,
)
from .base import clear_cuda_memory, run_async

logger = logging.getLogger(__name__)


class MultiAgentModule(dspy.Module):
    """
    DSPy Module wrapping a fairlib HierarchicalAgentRunner.

    Exposes only the final result to DSPy. The multi-agent coordination
    (manager delegating to workers) is internal.
    """

    def __init__(
        self,
        runner: "HierarchicalAgentRunner",
        input_field: str = "user_input",
        output_field: str = "response",
    ):
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
            self._manager_role = role_def.text if hasattr(role_def, "text") else str(role_def)
        except AttributeError:
            self._manager_role = "Coordinate workers to complete the task."
            logger.warning("Could not extract manager role, using default")

        # Extract worker info
        self._worker_roles = {}
        try:
            for name, worker in self.runner.workers.items():
                role_def = worker.planner.prompt_builder.role_definition
                self._worker_roles[name] = (
                    role_def.text if hasattr(role_def, "text") else str(role_def)
                )
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
            result = run_async(self.runner.arun(user_input))
        except Exception as e:
            logger.error(f"MultiAgent error: {e}")
            result = f"Error: {e}"

        # Extract string result
        if isinstance(result, str):
            response = result
        elif hasattr(result, "text"):
            response = result.text
        elif hasattr(result, "content"):
            response = result.content
        else:
            response = str(result)

        return dspy.Prediction(**{self.output_field: response})

    def get_manager_builder(self) -> Optional["PromptBuilder"]:
        """Get the manager's PromptBuilder."""
        try:
            return self.runner.manager.planner.prompt_builder
        except AttributeError:
            return None

    def get_worker_builders(self) -> Dict[str, "PromptBuilder"]:
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
        if hasattr(self.predict, "signature") and hasattr(self.predict.signature, "instructions"):
            if self.predict.signature.instructions:
                return self.predict.signature.instructions
        return self._manager_role

    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, "demos") and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, "_store") else demo
                demos.append(demo_dict)
        return demos


class MultiAgentOptimizer:
    """
    Optimizer for fairlib HierarchicalAgentRunner.

    Optimizes the manager and/or worker prompts while running
    the full multi-agent coordination pipeline.
    """

    def __init__(
        self,
        runner: "HierarchicalAgentRunner",
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
            from ..config import extract_multi_agent_config

            config_dict = extract_multi_agent_config(runner)
            self.config = OptimizedConfig(config=config_dict)

        self._module = None

    def compile(
        self,
        training_examples: List[TrainingExample],
        metric: Callable,
        worker_training_examples: Optional[Dict[str, List[TrainingExample]]] = None,
        worker_metrics: Optional[Dict[str, Callable]] = None,
        optimizer: str = "bootstrap",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        mipro_auto: Literal["light", "medium", "heavy"] = "light",
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
        - Use DSPy MIPROv2 to optimize role_definition/instructions
        - Uses manual bootstrap for examples (No DSPy)

        For fairlib agents, training data should reflect a run through the fair_llm pipe.

        Args:
            training_examples: Examples with full_trace showing complete manager workflow
            metric: Evaluation function (example, prediction, trace) -> bool/float
            worker_training_examples: Dict mapping worker names to their training examples.
                Each worker gets its own list of TrainingExamples. Only workers with
                training data will be optimized when optimize_workers=True.
            worker_metrics: Optional dict mapping worker names to custom metrics.
                If a worker is not in this dict, the default metric is used.
            optimizer: "bootstrap" or "mipro"
            max_bootstrapped_demos: Max demos to include
            max_labeled_demos: Max labeled demos (for MIPRO instruction candidates)
            mipro_auto: MIPROv2 intensity ('light', 'medium', 'heavy')
            training_data_path: Path to training data (for provenance)
            dspy_lm: DSPy LM for MIPROv2 instruction generation
            optimize_manager: Whether to optimize manager (defaults to instance setting)
            optimize_workers: Whether to optimize workers (defaults to instance setting)

        Returns:
            OptimizedConfig with optimized prompts for manager and workers
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
            logger.warning("Multi-agent optimization requires full_trace in training examples.")

        # Track starting state
        manager_prompts = self.config.config.get("manager", {}).get("prompts", {})
        examples_before = len(manager_prompts.get("examples", []))
        old_role = manager_prompts.get("role_definition", "")
        new_role = old_role

        # Run Bootstrap optimization
        logger.info(
            f"Running bootstrap to select full_trace examples ({len(training_examples)} candidates)"
        )
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

        # Update config
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

        # Worker optimization
        workers_optimized = []
        if optimize_workers and worker_training_examples:
            logger.info(f"Optimizing {len(worker_training_examples)} workers")
            optimized_worker_configs = self._optimize_workers(
                worker_training_examples=worker_training_examples,
                worker_metrics=worker_metrics,
                default_metric=metric,
                optimizer=optimizer,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                mipro_auto=mipro_auto,
                dspy_lm=dspy_lm,
            )

            # Ensure workers dict exists in config
            if "workers" not in self.config.config:
                self.config.config["workers"] = {}

            # Update config with optimized worker configs
            for worker_name, worker_config_dict in optimized_worker_configs.items():
                self.config.config["workers"][worker_name] = worker_config_dict
                workers_optimized.append(worker_name)

            logger.info(f"Workers optimized: {workers_optimized}")

        # Record provenance
        examples_after = len(
            self.config.config.get("manager", {}).get("prompts", {}).get("examples", [])
        )

        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, "__name__") else str(metric),
            training_data_path=training_data_path,
            training_data_hash=(
                compute_file_hash(training_data_path) if training_data_path else None
            ),
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
                "workers_optimized": workers_optimized,
            },
        )

        logger.info(
            f"Manager optimization complete. Examples: {examples_before} → {examples_after}"
        )
        if optimizer == "mipro":
            logger.info(f"Manager role definition changed: {new_role != old_role}")
        if workers_optimized:
            logger.info(f"Worker optimization complete for: {workers_optimized}")

        return self.config

    def _optimize_workers(
        self,
        worker_training_examples: Dict[str, List[TrainingExample]],
        worker_metrics: Optional[Dict[str, Callable]],
        default_metric: Callable,
        optimizer: str,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        mipro_auto: str,
        dspy_lm,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize each worker agent individually using AgentOptimizer.

        Args:
            worker_training_examples: Dict mapping worker names to training examples
            worker_metrics: Optional dict of per-worker metrics
            default_metric: Metric to use if worker not in worker_metrics
            optimizer: Optimizer type ("bootstrap" or "mipro")
            max_bootstrapped_demos: Max demos per worker
            max_labeled_demos: Max labeled demos for MIPRO
            mipro_auto: MIPRO intensity
            dspy_lm: DSPy LM for MIPRO instruction generation

        Returns:
            Dict mapping worker_name -> optimized config dict
        """
        from .agent import AgentOptimizer

        optimized_workers = {}

        for worker_name, worker in self.runner.workers.items():
            if worker_name not in worker_training_examples:
                logger.warning(
                    f"No training data for worker '{worker_name}', skipping optimization"
                )
                continue

            examples = worker_training_examples[worker_name]
            if not examples:
                logger.warning(
                    f"Empty training data for worker '{worker_name}', skipping optimization"
                )
                continue

            logger.info(f"Optimizing worker: {worker_name} ({len(examples)} examples)")

            # Use worker-specific metric if provided, otherwise default
            worker_metric = default_metric
            if worker_metrics and worker_name in worker_metrics:
                worker_metric = worker_metrics[worker_name]

            # Create optimizer for this worker
            worker_optimizer = AgentOptimizer(worker)

            try:
                worker_config = worker_optimizer.compile(
                    training_examples=examples,
                    metric=worker_metric,
                    optimizer=optimizer,
                    max_bootstrapped_demos=max_bootstrapped_demos,
                    max_labeled_demos=max_labeled_demos,
                    mipro_auto=mipro_auto,
                    dspy_lm=dspy_lm,
                )

                # Store the optimized config dict
                optimized_workers[worker_name] = worker_config.to_dict()
                logger.info(
                    f"Worker '{worker_name}' optimization complete: {len(worker_config.examples)} examples"
                )

            except Exception as e:
                logger.error(f"Failed to optimize worker '{worker_name}': {e}")
                continue

        return optimized_workers

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

            user_input = ex.inputs.get("user_input", "")

            try:
                # Run the full multi-agent system
                logger.debug(f"Running multi-agent on: {user_input[:50]}...")
                prediction = module(user_input=user_input)

                # Check metric
                dspy_example = dspy.Example(
                    user_input=user_input,
                    response=ex.expected_output,
                ).with_inputs("user_input")

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
    ) -> str:
        """
        Use MIPROv2 to optimize manager role_definition/instructions only.
        """
        import dspy
        from dspy.teleprompt import MIPROv2

        dspy.configure(lm=dspy_lm)

        module = MultiAgentModule(self.runner)
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
            new_role = optimized.get_optimized_manager_role()
            logger.info("MIPRO optimized manager role definition")
            return new_role
        except Exception as e:
            logger.warning(f"MIPRO instruction optimization failed: {e}")
            return (
                self.config.config.get("manager", {}).get("prompts", {}).get("role_definition", "")
            )

    def test(self, user_input: str) -> str:
        """Run a test input through the multi-agent system."""
        module = self._module or MultiAgentModule(self.runner)
        result = module(user_input=user_input)
        return result.response

    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)
