# fair_prompt_optimizer/optimizers/simple_llm.py
"""
SimpleLLM optimizer and DSPy module.

This module provides:
- SimpleLLMModule: DSPy module wrapping a simple LLM with system prompt
- SimpleLLMOptimizer: Optimizer for LLM + system prompt (no agent pipeline)

Use this when you just have an LLM with a system prompt and want
to optimize the prompt with few-shot examples.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import dspy

from ..config import (
    DSPyTranslator,
    OptimizedConfig,
    TrainingExample,
    compute_file_hash,
)
from .base import clear_cuda_memory

logger = logging.getLogger(__name__)


class SimpleLLMModule(dspy.Module):
    """
    DSPy Module wrapping a simple LLM with system prompt.

    No agent pipeline, just direct LLM calls.
    """

    def __init__(
        self,
        llm,
        system_prompt: str,
        input_field: str = "user_input",
        output_field: str = "response",
    ):
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
        """Signature is just the system prompt - MIPRO optimizes this directly."""
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

            if hasattr(response, "content"):
                result = response.content
            elif hasattr(response, "text"):
                result = response.text
            else:
                result = str(response)

        except Exception as e:
            logger.error(f"LLM error during forward pass: {e}", exc_info=True)
            result = f"Error: {e}"

        return dspy.Prediction(**{self.output_field: result})

    def get_optimized_prompt(self) -> str:
        """Get the optimized system prompt (instructions) if available."""
        if hasattr(self.predict, "signature") and hasattr(self.predict.signature, "instructions"):
            if self.predict.signature.instructions:
                return self.predict.signature.instructions
        return self.system_prompt

    def get_demos(self) -> List[Dict[str, str]]:
        """Extract demos from the predict object."""
        demos = []
        if hasattr(self.predict, "demos") and self.predict.demos:
            for demo in self.predict.demos:
                demo_dict = dict(demo) if hasattr(demo, "_store") else demo
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

            self.config = OptimizedConfig(
                config={
                    "version": "1.0",
                    "type": "simple_llm",
                    "prompts": {
                        "system_prompt": system_prompt,
                        "examples": [],
                    },
                    "model": model_info,
                }
            )

        self._module = None

    def _extract_model_info(self, llm) -> Dict[str, Any]:
        """Extract model information from LLM adapter."""
        adapter_name = type(llm).__name__

        # Try to get model name from common attributes
        model_name = (
            getattr(llm, "model_name", None)
            or getattr(llm, "model_id", None)
            or getattr(llm, "model", None)
            or "unknown"
        )

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
        mipro_auto: Literal["light", "medium", "heavy"] = "light",
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

        # Initialize the DSPy module
        module = SimpleLLMModule(self.llm, self.system_prompt)

        # Convert to DSPy format
        translator = DSPyTranslator(
            input_field=module.input_field, output_field=module.output_field
        )
        dspy_examples = translator.to_dspy_examples(training_examples)

        # Track starting state
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

        # Get demos DSPy selected
        demos = optimized_module.get_demos()

        if demos:
            demo_parts = [new_prompt, "\n# Examples:"]
            for demo in demos:
                inp = demo.get("user_input", "")
                out = demo.get("response", "")
                demo_parts.append(f"Input: {inp}\nOutput: {out}")
            new_prompt = "\n".join(demo_parts)

        # Update config
        self.config.config["prompts"]["system_prompt"] = new_prompt

        # Record provenance
        self.config.optimization.record_run(
            optimizer=optimizer,
            metric=metric.__name__ if hasattr(metric, "__name__") else str(metric),
            training_data_path=training_data_path,
            training_data_hash=(
                compute_file_hash(training_data_path) if training_data_path else None
            ),
            num_examples=len(training_examples),
            examples_after=len(demos),
            role_definition_changed=(new_prompt != old_prompt),
            optimizer_config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "mipro_auto": mipro_auto if optimizer == "mipro" else None,
            },
        )

        return self.config

    def test(self, user_input: str) -> str:
        """Run a test input through the LLM."""
        module = self._module or SimpleLLMModule(self.llm, self.system_prompt)
        result = module(user_input=user_input)
        return result.response

    def save(self, path: str):
        """Save the optimized config."""
        self.config.save(path)
