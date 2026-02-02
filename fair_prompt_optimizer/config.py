# fair_prompt_optimizer/config.py
"""
Configuration for FAIR Prompt Optimizer.

This module:
- IMPORTS core serialization from fairlib.utils.config_manager (for internal use)
- ADDS optimization provenance tracking

The config format is shared with fairlib, so:
- fairlib can load optimized configs directly
- fair_prompt_optimizer can optimize configs saved by fairlib
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fairlib.utils.config_manager import (
        extract_prompts,
        apply_prompts,
        extract_config,
        load_agent_config,
        extract_multi_agent_config,
    )

    from fairlib.core.prompts import PromptBuilder
    from fairlib.modules.agent.simple_agent import SimpleAgent

    FAIRLIB_AVAILABLE = True
except ImportError:
    FAIRLIB_AVAILABLE = False
    # Define stubs for when fairlib isn't available (e.g., testing)
    def extract_prompts(*args, **kwargs): raise ImportError("fairlib not installed")
    def apply_prompts(*args, **kwargs): raise ImportError("fairlib not installed")
    def extract_config(*args, **kwargs): raise ImportError("fairlib not installed")
    def load_agent_config(*args, **kwargs): raise ImportError("fairlib not installed")
    def save_agent_config(*args, **kwargs): raise ImportError("fairlib not installed")
    def extract_multi_agent_config(*args, **kwargs): raise ImportError("fairlib not installed")

logger = logging.getLogger(__name__)

@dataclass
class OptimizedPrompts:
    """
    Container for all optimizable prompt components.
    """
    role_definition: Optional[str] = None
    format_instructions: Optional[List[str]] = None
    
    # Track what actually changed
    role_definition_changed: bool = False
    format_instructions_changed: bool = False
    
    def has_changes(self) -> bool:
        """Check if any component was changed."""
        return self.role_definition_changed or self.format_instructions_changed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_definition": self.role_definition,
            "format_instructions": self.format_instructions,
            "role_definition_changed": self.role_definition_changed,
            "format_instructions_changed": self.format_instructions_changed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedPrompts':
        return cls(
            role_definition=data.get("role_definition"),
            format_instructions=data.get("format_instructions"),
            role_definition_changed=data.get("role_definition_changed", False),
            format_instructions_changed=data.get("format_instructions_changed", False),
        )

@dataclass
class OptimizationRun:
    """Record of a single optimization run."""
    timestamp: str
    optimizer: str
    metric: str
    training_data_hash: Optional[str] = None
    examples_before: int = 0
    examples_after: int = 0
    role_definition_changed: bool = False
    format_instructions_changed: bool = False
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRun':
        return cls(**data)


@dataclass
class OptimizationProvenance:
    """
    Tracks what conditions produced these prompts.
    
    This is fair_prompt_optimizer's value-add on top of fairlib configs.
    Only stores runs - other fields derived from runs list.
    """
    runs: List[OptimizationRun] = field(default_factory=list)
    
    @property
    def optimized(self) -> bool:
        return len(self.runs) > 0
    
    @property
    def optimizer(self) -> Optional[str]:
        return self.runs[-1].optimizer if self.runs else None
    
    @property
    def metric(self) -> Optional[str]:
        return self.runs[-1].metric if self.runs else None
    
    @property
    def created_at(self) -> Optional[str]:
        return self.runs[0].timestamp if self.runs else None
    
    @property
    def last_optimized_at(self) -> Optional[str]:
        return self.runs[-1].timestamp if self.runs else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": [run.to_dict() for run in self.runs],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationProvenance':
        return cls(
            runs=[OptimizationRun.from_dict(r) for r in data.get("runs", [])],
        )
    
    def record_run(
        self,
        optimizer: str,
        metric: str,
        training_data_path: Optional[str] = None,
        training_data_hash: Optional[str] = None,
        num_examples: int = 0,
        examples_before: int = 0,
        examples_after: int = 0,
        role_definition_changed: bool = False,
        format_instructions_changed: bool = False,
        optimizer_config: Optional[Dict[str, Any]] = None,
    ):
        """Record a new optimization run."""
        self.runs.append(OptimizationRun(
            timestamp=datetime.now().isoformat(),
            optimizer=optimizer,
            metric=metric,
            training_data_hash=training_data_hash,
            examples_before=examples_before,
            examples_after=examples_after,
            role_definition_changed=role_definition_changed,
            format_instructions_changed=format_instructions_changed,
            optimizer_config=optimizer_config or {},
        ))


@dataclass
class OptimizedConfig:
    """
    A fairlib config with optimization provenance attached.
    
    Structure:
    - config: The raw dict (compatible with fairlib's load_agent)
    - optimization: Provenance tracking (our value-add)
    """
    config: Dict[str, Any] = field(default_factory=dict)
    optimization: OptimizationProvenance = field(default_factory=OptimizationProvenance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result = dict(self.config)
        result["optimization"] = self.optimization.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedConfig':
        """Deserialize from dict."""
        data = dict(data)  # Copy to avoid mutation
        optimization_data = data.pop("optimization", {})
        return cls(
            config=data,
            optimization=OptimizationProvenance.from_dict(optimization_data),
        )
    
    @classmethod
    def from_agent(cls, agent: 'SimpleAgent') -> 'OptimizedConfig':
        """Extract config from a running agent using fairlib."""
        config = extract_config(agent)
        return cls(config=config)
    
    @classmethod 
    def from_file(cls, path: str) -> 'OptimizedConfig':
        """Load from JSON file."""
        data = load_agent_config(path)
        return cls.from_dict(data)
    
    # --- Convenience accessors ---

    @property
    def format_instructions(self) -> List[str]:
        return self.prompts.get("format_instructions", [])
    
    @format_instructions.setter
    def format_instructions(self, value: List[str]):
        if "prompts" not in self.config:
            self.config["prompts"] = {}
        self.config["prompts"]["format_instructions"] = value
    
    @property
    def prompts(self) -> Dict[str, Any]:
        return self.config.get("prompts", {})
    
    @prompts.setter
    def prompts(self, value: Dict[str, Any]):
        self.config["prompts"] = value
    
    @property
    def type(self) -> str:
        return self.config.get("type", "agent")
    
    @property
    def examples(self) -> List[str]:
        return self.prompts.get("examples", [])
    
    @examples.setter
    def examples(self, value: List[str]):
        if "prompts" not in self.config:
            self.config["prompts"] = {}
        self.config["prompts"]["examples"] = value
    
    @property
    def role_definition(self) -> Optional[str]:
        return self.prompts.get("role_definition")
    
    @role_definition.setter
    def role_definition(self, value: str):
        if "prompts" not in self.config:
            self.config["prompts"] = {}
        self.config["prompts"]["role_definition"] = value
    
    def get_prompt_builder(self) -> 'PromptBuilder':
        """Create a PromptBuilder from this config's prompts."""
        from fairlib.core.prompts import PromptBuilder
        builder = PromptBuilder()
        apply_prompts(self.prompts, builder)
        return builder
    
    def update_from_prompt_builder(self, builder: 'PromptBuilder'):
        """Update prompts from a PromptBuilder."""
        self.config["prompts"] = extract_prompts(builder)
    
    def save(self, path: str):
        """Save to JSON file."""
        save_optimized_config(self, path)


def load_optimized_config(path: str) -> OptimizedConfig:
    """
    Load a config file as OptimizedConfig.
    
    Works with:
    - Configs saved by fairlib (no optimization section)
    - Configs saved by fair_prompt_optimizer (with optimization)
    """
    return OptimizedConfig.from_file(path)


def save_optimized_config(config: OptimizedConfig, path: str):
    """Save OptimizedConfig to JSON file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"Saved optimized config to {filepath}")


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file for provenance tracking."""
    filepath = Path(path)
    if not filepath.exists():
        return ""
    with open(filepath, 'rb') as f:
        return f"sha256:{hashlib.sha256(f.read()).hexdigest()[:16]}"


@dataclass
class TrainingExample:
    """A single training example for optimization."""
    inputs: Dict[str, str]
    expected_output: str
    full_trace: Optional[str] = None  # Full conversation trace for few-shot examples
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"inputs": self.inputs, "expected_output": self.expected_output}
        if self.full_trace:
            result["full_trace"] = self.full_trace
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        return cls(
            inputs=data.get("inputs", {}),
            expected_output=data.get("expected_output", ""),
            full_trace=data.get("full_trace"),
        )


def load_training_examples(path: str) -> List[TrainingExample]:
    """Load training examples from JSON file."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Training data not found: {filepath}")
    with open(filepath, 'r') as f:
        return [TrainingExample.from_dict(item) for item in json.load(f)]


def save_training_examples(examples: List[TrainingExample], path: str):
    """Save training examples to JSON file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump([ex.to_dict() for ex in examples], f, indent=2)


class DSPyTranslator:
    """Translate between TrainingExamples and DSPy format."""
    
    def __init__(self, input_field: str = "user_input", output_field: str = "expected_output"):
        self.input_field = input_field
        self.output_field = output_field
    
    def to_dspy_examples(self, examples: List[TrainingExample]) -> List:
        """Convert TrainingExamples to DSPy Examples."""
        import dspy
        
        dspy_examples = []
        for ex in examples:
            user_input = ex.inputs.get(self.input_field, ex.inputs.get("user_input", ""))
            dspy_ex = dspy.Example(
                **{self.input_field: user_input},
                **{self.output_field: ex.expected_output},
            ).with_inputs(self.input_field)
            dspy_examples.append(dspy_ex)
        
        return dspy_examples