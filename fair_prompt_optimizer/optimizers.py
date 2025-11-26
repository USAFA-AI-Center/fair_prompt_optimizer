"""
optimizers.py
=============

Wraps DSPy's optimization algorithms with a FAIR-LLM friendly interface.

This module provides the FAIRPromptOptimizer class which handles:
1. Loading FAIR-LLM JSON configurations
2. Translating to DSPy format
3. Running optimization (BootstrapFewShot, MIPROv2)
4. Extracting results back to FAIR-LLM format
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

# MIPROv2 may not be available in all DSPy versions
try:
    from dspy.teleprompt import MIPROv2
    HAS_MIPRO = True
except ImportError:
    HAS_MIPRO = False
    MIPROv2 = None

from .translator import (
    DSPyTranslator,
    FAIRConfig,
    TrainingExample,
    load_fair_config,
    save_fair_config,
    load_training_examples,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Results from an optimization run.
    
    Attributes:
        config: The optimized FAIR-LLM configuration
        score: Best score achieved during optimization
        optimizer_name: Name of the optimizer used
        num_trials: Number of optimization trials run
        duration_seconds: Time taken for optimization
        output_path: Path where the optimized config was saved
    """
    config: FAIRConfig
    score: Optional[float] = None
    optimizer_name: str = ""
    num_trials: Optional[int] = None
    duration_seconds: Optional[float] = None
    output_path: Optional[str] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class FAIRPromptOptimizer:
    """
    Main optimizer class that bridges FAIR-LLM and DSPy.
    
    This class provides a simple interface for optimizing FAIR-LLM prompts
    using DSPy's optimization algorithms without requiring users to understand
    DSPy internals.
    """
    
    def __init__(
        self,
        lm_model: str = "",
        api_key: Optional[str] = None,
        **lm_kwargs
    ):
        self.lm_model = lm_model
        
        # Initialize the LM
        lm_config = {"model": lm_model}
        if api_key:
            lm_config["api_key"] = api_key
        lm_config.update(lm_kwargs)
        
        self.lm = dspy.LM(**lm_config)
        dspy.configure(lm=self.lm)
        
        # Initialize translator
        self.translator = DSPyTranslator()
        
        logger.info(f"Initialized FAIRPromptOptimizer with model: {lm_model}")
    
    def optimize_bootstrap(
        self,
        fair_config_path: Union[str, Path],
        training_examples: Union[List[TrainingExample], str, Path],
        metric: Callable,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        max_rounds: int = 1,
        max_errors: int = 5,
        output_path: Optional[Union[str, Path]] = None,
        use_random_search: bool = False,
        num_candidate_programs: int = 10,
        use_chain_of_thought: bool = True,
    ) -> OptimizationResult:
        """
        Run BootstrapFewShot optimization.
        
        This optimizer generates high-quality few-shot examples by running
        your program on training data and keeping examples that pass your metric.
        """
        start_time = datetime.now()
        
        # Load configuration and training data
        config = load_fair_config(fair_config_path)
        
        if isinstance(training_examples, (str, Path)):
            training_examples = load_training_examples(training_examples)
        
        logger.info(f"Loaded config from {fair_config_path}")
        logger.info(f"Training on {len(training_examples)} examples")
        
        # Convert to DSPy format
        module = self.translator.config_to_module(config, use_chain_of_thought)
        dspy_examples = self.translator.training_examples_to_dspy(training_examples)
        
        # Wrap the metric to handle FAIR format
        wrapped_metric = self._wrap_metric(metric)
        
        # Create optimizer
        optimizer_config = {
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "max_rounds": max_rounds,
            "max_errors": max_errors,
        }
        
        if use_random_search:
            optimizer = BootstrapFewShotWithRandomSearch(
                metric=wrapped_metric,
                num_candidate_programs=num_candidate_programs,
                **optimizer_config
            )
            optimizer_name = "bootstrap_random_search"
        else:
            optimizer = BootstrapFewShot(
                metric=wrapped_metric,
                **optimizer_config
            )
            optimizer_name = "bootstrap"
        
        logger.info(f"Running {optimizer_name} optimization...")
        
        # Run optimization
        optimized_module = optimizer.compile(
            module,
            trainset=dspy_examples
        )
        
        # Extract optimized config
        optimized_config = self.translator.extract_optimized_config(
            original_config=config,
            optimized_module=optimized_module,
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config
        )
        
        # Save if output path provided
        if output_path:
            save_fair_config(optimized_config, output_path)
            logger.info(f"Saved optimized config to {output_path}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            config=optimized_config,
            optimizer_name=optimizer_name,
            duration_seconds=duration,
            output_path=str(output_path) if output_path else None
        )
    
    def optimize_mipro(
        self,
        fair_config_path: Union[str, Path],
        training_examples: Union[List[TrainingExample], str, Path],
        metric: Callable,
        auto: str = "light",
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        num_trials: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        use_chain_of_thought: bool = True,
        prompt_model: Optional[str] = None,
        requires_permission_to_run: bool = False,
    ) -> OptimizationResult:
        """
        Run MIPROv2 optimization.
        
        This optimizer jointly optimizes both instructions AND few-shot examples
        using Bayesian optimization. It's more powerful but requires more compute.
        """
        if not HAS_MIPRO:
            raise ImportError(
                "MIPROv2 is not available in your DSPy version. "
                "Please upgrade: pip install --upgrade dspy"
            )
        
        start_time = datetime.now()
        
        # Load configuration and training data
        config = load_fair_config(fair_config_path)
        
        if isinstance(training_examples, (str, Path)):
            training_examples = load_training_examples(training_examples)
        
        logger.info(f"Loaded config from {fair_config_path}")
        logger.info(f"Training on {len(training_examples)} examples")
        
        # Convert to DSPy format
        module = self.translator.config_to_module(config, use_chain_of_thought)
        dspy_examples = self.translator.training_examples_to_dspy(training_examples)
        
        # Wrap the metric
        wrapped_metric = self._wrap_metric(metric)
        
        # Create optimizer
        mipro_kwargs = {
            "metric": wrapped_metric,
            "auto": auto,
        }
        
        if prompt_model:
            prompt_lm = dspy.LM(prompt_model)
            mipro_kwargs["prompt_model"] = prompt_lm
        
        optimizer = MIPROv2(**mipro_kwargs)
        optimizer_name = f"mipro_{auto}"
        
        logger.info(f"Running MIPROv2 ({auto}) optimization...")
        
        # Build compile kwargs
        compile_kwargs = {
            "trainset": dspy_examples,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "requires_permission_to_run": requires_permission_to_run,
        }
        
        if num_trials:
            compile_kwargs["num_trials"] = num_trials
        
        # Run optimization
        optimized_module = optimizer.compile(
            module.deepcopy(),
            **compile_kwargs
        )
        
        # Extract optimized config
        optimizer_config = {
            "auto": auto,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "num_trials": num_trials,
        }
        
        optimized_config = self.translator.extract_optimized_config(
            original_config=config,
            optimized_module=optimized_module,
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config
        )
        
        # Save if output path provided
        if output_path:
            save_fair_config(optimized_config, output_path)
            logger.info(f"Saved optimized config to {output_path}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            config=optimized_config,
            optimizer_name=optimizer_name,
            duration_seconds=duration,
            output_path=str(output_path) if output_path else None
        )
    
    def evaluate(
        self,
        fair_config_path: Union[str, Path],
        test_examples: Union[List[TrainingExample], str, Path],
        metric: Callable,
        use_chain_of_thought: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a FAIR-LLM configuration on test examples.
        
        Useful for comparing before/after optimization performance.
        """
        config = load_fair_config(fair_config_path)
        
        if isinstance(test_examples, (str, Path)):
            test_examples = load_training_examples(test_examples)
        
        module = self.translator.config_to_module(config, use_chain_of_thought)
        dspy_examples = self.translator.training_examples_to_dspy(test_examples)
        wrapped_metric = self._wrap_metric(metric)
        
        # Run evaluation
        evaluator = dspy.Evaluate(
            devset=dspy_examples,
            metric=wrapped_metric,
            num_threads=1,
            display_progress=True
        )
        
        score = evaluator(module)
        
        return {
            "score": score,
            "num_examples": len(test_examples),
            "config_path": str(fair_config_path),
            "optimized": config.metadata.optimized
        }
    
    def _wrap_metric(self, metric: Callable) -> Callable:
        """
        Wrap a user-provided metric to work with DSPy's format.
        
        The wrapper handles the translation between DSPy's (example, prediction, trace)
        format and the user's expected format.
        """
        def wrapped(example, prediction, trace=None):
            try:
                return metric(example, prediction, trace)
            except Exception as e:
                logger.warning(f"Metric evaluation failed: {e}")
                return 0.0
        
        return wrapped


def create_optimizer(
    model: str = "openai/gpt-4o-mini",
    **kwargs
) -> FAIRPromptOptimizer:
    """
    Factory function to create an optimizer instance.
    
    Args:
        model: LM model string
        **kwargs: Additional arguments for FAIRPromptOptimizer
        
    Returns:
        Configured FAIRPromptOptimizer instance
    """
    return FAIRPromptOptimizer(lm_model=model, **kwargs)
