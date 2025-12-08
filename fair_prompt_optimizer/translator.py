# translator.py

from dataclasses import dataclass, field, asdict
from datetime import datetime
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
class FAIRConfig:
    """
    Complete FAIR-LLM PromptBuilder configuration.
    
    This is the JSON contract between FAIR-LLM and the optimizer.
    """
    version: str = "1.0"
    role_definition: Optional[str] = None
    tool_instructions: List[ToolInstruction] = field(default_factory=list)
    worker_instructions: List[WorkerInstruction] = field(default_factory=list)
    format_instructions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
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
            metadata=OptimizationMetadata.from_dict(data.get("metadata", {}))
        )

def load_fair_config(path: str | Path) -> FAIRConfig:
    """
    Load a FAIR-LLM configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        FAIRConfig object
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return FAIRConfig.from_dict(data)


def save_fair_config(config: FAIRConfig, path: str | Path) -> None:
    """
    Save a FAIR-LLM configuration to a JSON file.
    
    Args:
        config: FAIRConfig object to save
        path: Destination path for the JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_training_examples(path: str | Path) -> List[TrainingExample]:
    """
    Load training examples from a JSON file.
    
    Expected format:
    [
        {"inputs": {"user_query": "..."}, "expected_output": "..."},
        ...
    ]
    
    Args:
        path: Path to JSON file containing training examples
        
    Returns:
        List of TrainingExample objects
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [TrainingExample.from_dict(ex) for ex in data]


def save_training_examples(examples: List[TrainingExample], path: str | Path) -> None:
    """Save training examples to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump([ex.to_dict() for ex in examples], f, indent=2)


class DSPyTranslator:
    """
    Translates between FAIR-LLM configurations and DSPy modules.
    
    This class handles the conversion of FAIR-LLM's PromptBuilder JSON format
    into DSPy Signatures and Modules for optimization, and the reverse
    translation of optimized DSPy state back to FAIR-LLM format.
    """
    
    def __init__(self, input_field: str = "user_input", output_field: str = "response"):
        """
        Initialize the translator.
        
        Args:
            input_field: Name of the input field in DSPy signature
            output_field: Name of the output field in DSPy signature
        """
        self.input_field = input_field
        self.output_field = output_field
    
    def config_to_signature(self, config: FAIRConfig) -> Type[dspy.Signature]:
        """
        Create a DSPy Signature class from a FAIR-LLM configuration.
        
        The signature's docstring is constructed from the role_definition
        and format_instructions, which is what DSPy optimizers will modify.
        
        Args:
            config: FAIR-LLM configuration
            
        Returns:
            A dynamically created DSPy Signature class
        """
        # Build the docstring from role definition and format instructions
        docstring_parts = []
        
        if config.role_definition:
            docstring_parts.append(config.role_definition)
        
        if config.format_instructions:
            docstring_parts.append("\nFormat Instructions:")
            for fmt in config.format_instructions:
                docstring_parts.append(f"- {fmt}")
        
        docstring = "\n".join(docstring_parts) if docstring_parts else "Complete the given task."
        
        # Create the signature class dynamically
        # We use type() to create a new class with the proper docstring
        signature_dict = {
            "__doc__": docstring,
            "__annotations__": {
                self.input_field: str,
                self.output_field: str,
            },
            self.input_field: dspy.InputField(desc="User's request or query"),
            self.output_field: dspy.OutputField(desc="Agent's response to the user"),
        }
        
        FAIRSignature = type("FAIRSignature", (dspy.Signature,), signature_dict)
        
        return FAIRSignature
    
    def config_to_module(
        self, 
        config: FAIRConfig, 
        use_chain_of_thought: bool = True
    ) -> dspy.Module:
        """
        Create a DSPy Module from a FAIR-LLM configuration.
        
        Args:
            config: FAIR-LLM configuration
            use_chain_of_thought: If True, use ChainOfThought; otherwise use Predict
            
        Returns:
            A DSPy Module (ChainOfThought or Predict)
        """
        signature = self.config_to_signature(config)
        
        if use_chain_of_thought:
            module = dspy.ChainOfThought(signature)
        else:
            module = dspy.Predict(signature)
        
        # If we have existing examples, convert them to demos
        if config.examples:
            demos = self._examples_to_demos(config.examples)
            module.demos = demos
        
        return module
    
    def training_examples_to_dspy(
        self, 
        examples: List[TrainingExample]
    ) -> List[dspy.Example]:
        """
        Convert FAIR training examples to DSPy Example format.
        
        Args:
            examples: List of TrainingExample objects
            
        Returns:
            List of dspy.Example objects ready for optimization
        """
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
    
    def extract_optimized_config(
        self,
        original_config: FAIRConfig,
        optimized_module: dspy.Module,
        optimizer_name: str,
        optimizer_config: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None
    ) -> FAIRConfig:
        """
        Extract the optimized state from a DSPy module and create a new FAIR config.
        
        This is the reverse translation: DSPy optimized state -> FAIR-LLM JSON format.
        
        Handles both legacy DSPy (<2.5) and modern DSPy (2.5+) state structures.
        """
        # Start with a copy of the original config
        optimized = FAIRConfig(
            version=original_config.version,
            tool_instructions=original_config.tool_instructions.copy(),
            worker_instructions=original_config.worker_instructions.copy(),
        )
        
        # Get the full state
        state = optimized_module.dump_state()
        
        # Extract optimized instructions and demos
        # DSPy 2.5+ nests these under predictor names (e.g., 'predict')
        demos = []
        instructions = None
        
        # Strategy 1: Try legacy format (top-level demos)
        if "demos" in state and state["demos"]:
            demos = state["demos"]
            
        # Strategy 2: Try DSPy 2.5+ format (nested under predictor names)
        if not demos:
            for key, value in state.items():
                if isinstance(value, dict):
                    # Check for demos in this predictor's state
                    if "demos" in value and value["demos"]:
                        demos = value["demos"]
                    # Check for instructions in signature
                    if "signature" in value and isinstance(value["signature"], dict):
                        if "instructions" in value["signature"]:
                            instructions = value["signature"]["instructions"]
                    # Break after finding first predictor with demos
                    if demos:
                        break
        
        # Strategy 3: Try accessing predictors directly (fallback)
        if not demos:
            try:
                for name, predictor in optimized_module.named_predictors():
                    if hasattr(predictor, 'demos') and predictor.demos:
                        # Convert Example objects to dicts
                        demos = [dict(d) for d in predictor.demos]
                        break
            except Exception:
                pass
        
        # Extract instructions
        # Strategy 1: From state (found above)
        # Strategy 2: From module.signature (legacy)
        if not instructions:
            if hasattr(optimized_module, 'signature') and optimized_module.signature:
                if hasattr(optimized_module.signature, 'instructions'):
                    instructions = optimized_module.signature.instructions
        
        # Strategy 3: From predictor signature
        if not instructions:
            try:
                for name, predictor in optimized_module.named_predictors():
                    if hasattr(predictor, 'signature') and predictor.signature:
                        if hasattr(predictor.signature, 'instructions'):
                            instructions = predictor.signature.instructions
                            break
            except Exception:
                pass
        
        # Set role definition (use extracted or fall back to original)
        if instructions:
            optimized.role_definition = instructions
        else:
            optimized.role_definition = original_config.role_definition
        
        # Preserve format instructions (these aren't optimized by DSPy)
        optimized.format_instructions = original_config.format_instructions.copy()
        
        # Convert demos to FAIR-LLM example format
        optimized.examples = []
        for demo in demos:
            # Handle both dict and Example objects
            if hasattr(demo, '_store'):
                demo = dict(demo)
            example_text = self._demo_to_example_text(demo)
            if example_text:
                optimized.examples.append(example_text)
        
        # Update metadata
        optimized.metadata = OptimizationMetadata(
            optimized=True,
            optimized_at=datetime.now().isoformat(),
            optimizer=optimizer_name,
            optimizer_config=optimizer_config,
            score=score,
            num_training_examples=len(demos) if demos else None
        )
        
        return optimized
    
    def _examples_to_demos(self, examples: List[str]) -> List[Dict[str, str]]:
        """
        Convert FAIR-LLM example strings to DSPy demo format.
        
        Args:
            examples: List of example strings in FAIR-LLM format
            
        Returns:
            List of demo dictionaries for DSPy
        """
        demos = []
        
        for example_text in examples:
            demo = self._parse_example_text(example_text)
            if demo:
                demos.append(demo)
        
        return demos
    
    def _parse_example_text(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse a FAIR-LLM example string into a demo dictionary.
        
        Expected format:
            User: <input>
            Thought: <reasoning>  (optional)
            Response: <output>
        
        Args:
            text: Example string
            
        Returns:
            Dictionary with parsed fields, or None if parsing fails
        """
        demo = {}
        
        lines = text.strip().split('\n')
        current_key = None
        current_value = []
        
        key_mapping = {
            "user:": self.input_field,
            "thought:": "reasoning",
            "response:": self.output_field,
            "action:": "action",
            "observation:": "observation",
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a new field
            found_key = None
            for prefix, field_name in key_mapping.items():
                if line_lower.startswith(prefix):
                    # Save previous field
                    if current_key and current_value:
                        demo[current_key] = '\n'.join(current_value).strip()
                    
                    current_key = field_name
                    current_value = [line[len(prefix):].strip()]
                    found_key = True
                    break
            
            if not found_key and current_key:
                current_value.append(line)
        
        # Save last field
        if current_key and current_value:
            demo[current_key] = '\n'.join(current_value).strip()
        
        return demo if demo else None
    
    def _demo_to_example_text(self, demo: Dict[str, Any]) -> str:
        """
        Convert a DSPy demo dictionary to FAIR-LLM example text format.
        
        Args:
            demo: Dictionary with demo fields
            
        Returns:
            Formatted example string
        """
        parts = []
        
        # Input field
        input_val = demo.get(self.input_field, demo.get("user_input", ""))
        if input_val:
            parts.append(f"User: {input_val}")
        
        # Reasoning (if present, from ChainOfThought)
        reasoning = demo.get("reasoning", demo.get("rationale", ""))
        if reasoning:
            parts.append(f"Thought: {reasoning}")
        
        # Output field
        output_val = demo.get(self.output_field, demo.get("response", demo.get("answer", "")))
        if output_val:
            parts.append(f"Response: {output_val}")
        
        return '\n'.join(parts)