# translator.py
"""
Handles bidirectional translation between FAIR-LLM's PromptBuilder JSON format
and DSPy's Module/Signature system.

The JSON format serves as the contract between FAIR-LLM and this optimizer,
ensuring FAIR-LLM never needs to import DSPy.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import json

import dspy

@dataclass
class TrainingExample:
    """
    A single training example for optimization.
    
    Attributes:
        inputs: Dictionary of input fields (e.g., {"user_query": "What is 2+2?"})
        expected_output: The expected response string
        metadata: Optional metadata about this example
    """
    inputs: Dict[str, str]
    expected_output: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """Create from dictionary."""
        return cls(
            inputs=data["inputs"],
            expected_output=data["expected_output"],
            metadata=data.get("metadata")
        )


@dataclass
class ToolInstruction:
    """Represents a tool available to the agent."""
    name: str
    description: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ToolInstruction":
        return cls(name=data["name"], description=data["description"])


@dataclass
class WorkerInstruction:
    """Represents a worker agent available for delegation."""
    name: str
    role_description: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "role_description": self.role_description}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "WorkerInstruction":
        return cls(name=data["name"], role_description=data["role_description"])


@dataclass
class OptimizationMetadata:
    """Metadata about the optimization process."""
    optimized: bool = False
    optimized_at: Optional[str] = None
    optimizer: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    num_training_examples: Optional[int] = None
    original_config_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class ModelConfig:
    """Configuration for the LLM."""
    model_name: str = "dolphin3-qwen25-3b"
    adapter: str = "HuggingFaceAdapter"  # "HuggingFaceAdapter", "OpenAIAdapter", etc.
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional kwargs for adapter
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "adapter": self.adapter,
            "adapter_kwargs": self.adapter_kwargs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            model_name=data.get("model_name", "dolphin3-qwen25-3b"),
            adapter=data.get("adapter", "HuggingFaceAdapter"),
            adapter_kwargs=data.get("adapter_kwargs", {})
        )
    
@dataclass
class AgentConfig:
    """Configuration for the agent structure."""
    agent_type: str = "SimpleAgent"  # "SimpleAgent", future: "MultiAgentManager"
    planner_type: str = "SimpleReActPlanner" # future: any planners that we add to fair_llm
    max_steps: int = 10
    tools: List[str] = field(default_factory=list)  # Tool class names: ["SafeCalculatorTool"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "planner_type": self.planner_type,
            "max_steps": self.max_steps,
            "tools": self.tools
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(
            agent_type=data.get("agent_type", "SimpleAgent"),
            planner_type=data.get("planner_type", "SimpleReActPlanner"),
            max_steps=data.get("max_steps", 10),
            tools=data.get("tools", [])
        )

@dataclass
class FAIRConfig:
    """
    Complete FAIR-LLM agent configuration.
    
    This is the JSON contract between FAIR-LLM and the optimizer.
    Contains everything needed to recreate an optimized agent.
    """
    # Version for backwards compatibility
    version: str = "1.0"
    
    # Prompt configuration
    role_definition: Optional[str] = None
    tool_instructions: List[ToolInstruction] = field(default_factory=list)
    worker_instructions: List[WorkerInstruction] = field(default_factory=list)
    format_instructions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Optimization metadata
    metadata: OptimizationMetadata = field(default_factory=OptimizationMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "version": self.version,
            "role_definition": self.role_definition,
            "tool_instructions": [t.to_dict() for t in self.tool_instructions],
            "worker_instructions": [w.to_dict() for w in self.worker_instructions],
            "format_instructions": self.format_instructions,
            "examples": self.examples,
            "model": self.model.to_dict(),
            "agent": self.agent.to_dict(),
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FAIRConfig":
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            role_definition=data.get("role_definition"),
            tool_instructions=[
                ToolInstruction.from_dict(t) 
                for t in data.get("tool_instructions", [])
            ],
            worker_instructions=[
                WorkerInstruction.from_dict(w) 
                for w in data.get("worker_instructions", [])
            ],
            format_instructions=data.get("format_instructions", []),
            examples=data.get("examples", []),
            model=ModelConfig.from_dict(data.get("model", {})),
            agent=AgentConfig.from_dict(data.get("agent", {})),
            metadata=OptimizationMetadata.from_dict(data.get("metadata", {}))
        )

class DSPyTranslator:
    """Translates between FAIR-LLM JSON format and DSPy modules."""
    def __init__(self, input_field: str = "user_input", output_field: str = "response"):
        """
        Initialize the translator.
        
        Args:
            input_field: Name of the input field in DSPy signature
            output_field: Name of the output field in DSPy signature
        """
        self.input_field = input_field
        self.output_field = output_field
    
    def training_examples_to_dspy(self, examples: List[TrainingExample]) -> List[dspy.Example]:
        """Convert FAIR training examples to DSPy Example format"""
        dspy_examples = []
        
        for ex in examples:
            primary_input = ex.inputs.get(self.input_field, "")
            if not primary_input:
                # Fall back to common field names
                primary_input = ex.inputs.get("user_query", ex.inputs.get("query", ""))
            if not primary_input:
                # Fall back to first input value
                primary_input = next(iter(ex.inputs.values()), "")
            
            dspy_ex = dspy.Example(**{
                self.input_field: primary_input,
                self.output_field: ex.expected_output
            }).with_inputs(self.input_field)
            
            dspy_examples.append(dspy_ex)
        
        return dspy_examples
    
    def _examples_to_demos(self, examples: List[str]) -> List[dspy.Example]:
        """Convert FAIR example strings to DSPy demo Examples."""
        demos = []
        for example_text in examples:
            demo = self._parse_example_text(example_text)
            if demo:
                dspy_demo = dspy.Example(**demo).with_inputs(self.input_field)
                demos.append(dspy_demo)
        return demos
    
    def _parse_example_text(self, text: str) -> Optional[Dict[str, str]]:
        """Parse a FAIR-LLM example string into a demo dictionary."""
        result = {}
        
        # Map of prefixes to field names
        prefix_map = {
            "user:": self.input_field,
            "thought:": "reasoning",
            "response:": self.output_field,
        }
        
        current_field = None
        current_content = []
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            
            # Check if line starts a new field
            matched = False
            for prefix, field_name in prefix_map.items():
                if line_lower.startswith(prefix):
                    # Save previous field
                    if current_field and current_content:
                        result[current_field] = '\n'.join(current_content).strip()
                    
                    # Start new field
                    current_field = field_name
                    content = line[len(prefix):].strip()
                    current_content = [content] if content else []
                    matched = True
                    break
            
            if not matched and current_field:
                current_content.append(line)
        
        # Save last field
        if current_field and current_content:
            result[current_field] = '\n'.join(current_content).strip()
        
        return result if result else None
    
    def _demo_to_example_text(self, demo: Dict[str, str]) -> str:
        """Convert a demo dictionary back to FAIR example text format."""
        parts = []
        
        input_val = demo.get(self.input_field, demo.get("user_input", ""))
        if input_val:
            parts.append(f"User: {input_val}")
        
        reasoning_val = demo.get("reasoning", "")
        if reasoning_val:
            parts.append(f"Thought: {reasoning_val}")
        
        output_val = demo.get(self.output_field, demo.get("response", demo.get("answer", "")))
        if output_val:
            parts.append(f"Response: {output_val}")
        
        return '\n'.join(parts)
    
# =============================================================================
# File I/O Functions
# =============================================================================

def load_fair_config(path: str) -> FAIRConfig:
    """Load a FAIRConfig from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return FAIRConfig.from_dict(data)


def save_fair_config(config: FAIRConfig, path: str) -> None:
    """Save a FAIRConfig to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_training_examples(path: str) -> List[TrainingExample]:
    """Load training examples from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [TrainingExample.from_dict(ex) for ex in data]


def save_training_examples(examples: List[TrainingExample], path: str) -> None:
    """Save training examples to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump([ex.to_dict() for ex in examples], f, indent=2)