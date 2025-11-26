"""
test_translator.py

Tests for the FAIR-LLM <-> DSPy translation layer.
"""

import pytest
from pathlib import Path
import tempfile

from fair_prompt_optimizer.translator import (
    TrainingExample,
    ToolInstruction,
    WorkerInstruction,
    FAIRConfig,
    OptimizationMetadata,
    DSPyTranslator,
    load_fair_config,
    save_fair_config,
    load_training_examples,
    save_training_examples,
)


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""
    
    def test_creation(self):
        example = TrainingExample(
            inputs={"user_query": "What is 2+2?"},
            expected_output="4"
        )
        assert example.inputs["user_query"] == "What is 2+2?"
        assert example.expected_output == "4"
    
    def test_to_dict(self):
        example = TrainingExample(
            inputs={"user_query": "test"},
            expected_output="result",
            metadata={"source": "manual"}
        )
        d = example.to_dict()
        assert d["inputs"]["user_query"] == "test"
        assert d["expected_output"] == "result"
        assert d["metadata"]["source"] == "manual"
    
    def test_from_dict(self):
        data = {
            "inputs": {"user_query": "hello"},
            "expected_output": "world"
        }
        example = TrainingExample.from_dict(data)
        assert example.inputs["user_query"] == "hello"
        assert example.expected_output == "world"


class TestFAIRConfig:
    """Tests for FAIRConfig dataclass."""
    
    def test_empty_config(self):
        config = FAIRConfig()
        assert config.version == "1.0"
        assert config.role_definition is None
        assert config.tool_instructions == []
        assert config.examples == []
    
    def test_full_config(self):
        config = FAIRConfig(
            role_definition="You are a helpful assistant.",
            tool_instructions=[
                ToolInstruction(name="calculator", description="Does math")
            ],
            format_instructions=["Be concise"],
            examples=["User: Hi\nResponse: Hello!"]
        )
        assert config.role_definition == "You are a helpful assistant."
        assert len(config.tool_instructions) == 1
        assert config.tool_instructions[0].name == "calculator"
    
    def test_to_dict_roundtrip(self):
        original = FAIRConfig(
            role_definition="Test role",
            tool_instructions=[
                ToolInstruction(name="tool1", description="desc1")
            ],
            format_instructions=["format1", "format2"],
            examples=["example1"]
        )
        
        d = original.to_dict()
        restored = FAIRConfig.from_dict(d)
        
        assert restored.role_definition == original.role_definition
        assert len(restored.tool_instructions) == len(original.tool_instructions)
        assert restored.format_instructions == original.format_instructions
        assert restored.examples == original.examples


class TestFileIO:
    """Tests for file I/O functions."""
    
    def test_save_and_load_config(self):
        config = FAIRConfig(
            role_definition="Test agent",
            examples=["example1", "example2"]
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            save_fair_config(config, path)
            loaded = load_fair_config(path)
            
            assert loaded.role_definition == config.role_definition
            assert loaded.examples == config.examples
        finally:
            Path(path).unlink()
    
    def test_save_and_load_training_examples(self):
        examples = [
            TrainingExample(inputs={"q": "1"}, expected_output="a"),
            TrainingExample(inputs={"q": "2"}, expected_output="b"),
        ]
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            save_training_examples(examples, path)
            loaded = load_training_examples(path)
            
            assert len(loaded) == 2
            assert loaded[0].inputs["q"] == "1"
            assert loaded[1].expected_output == "b"
        finally:
            Path(path).unlink()


class TestDSPyTranslator:
    """Tests for the DSPy translation layer."""
    
    def test_config_to_signature(self):
        config = FAIRConfig(
            role_definition="You are a math tutor.",
            format_instructions=["Show your work", "Be precise"]
        )
        
        translator = DSPyTranslator()
        signature = translator.config_to_signature(config)
        
        # Check that signature was created with proper docstring
        assert "math tutor" in signature.__doc__
        assert "Show your work" in signature.__doc__
    
    def test_training_examples_to_dspy(self):
        examples = [
            TrainingExample(
                inputs={"user_query": "What is 2+2?"},
                expected_output="4"
            ),
            TrainingExample(
                inputs={"user_query": "Hello"},
                expected_output="Hi there!"
            ),
        ]
        
        translator = DSPyTranslator()
        dspy_examples = translator.training_examples_to_dspy(examples)
        
        assert len(dspy_examples) == 2
        assert dspy_examples[0].user_input == "What is 2+2?"
        assert dspy_examples[0].response == "4"
    
    def test_parse_example_text(self):
        translator = DSPyTranslator()
        
        text = """User: What is the capital of France?
Thought: I need to recall geography facts.
Response: Paris"""
        
        demo = translator._parse_example_text(text)
        
        assert demo["user_input"] == "What is the capital of France?"
        assert demo["reasoning"] == "I need to recall geography facts."
        assert demo["response"] == "Paris"
    
    def test_demo_to_example_text(self):
        translator = DSPyTranslator()
        
        demo = {
            "user_input": "Hello",
            "reasoning": "Greeting detected",
            "response": "Hi!"
        }
        
        text = translator._demo_to_example_text(demo)
        
        assert "User: Hello" in text
        assert "Thought: Greeting detected" in text
        assert "Response: Hi!" in text


class TestMetrics:
    """Tests for metric functions."""
    
    def test_exact_match(self):
        from fair_prompt_optimizer.metrics import exact_match
        
        class MockExample:
            response = "hello"
        
        class MockPrediction:
            response = "hello"
        
        assert exact_match(MockExample(), MockPrediction()) is True
        
        MockPrediction.response = "HELLO"  # Case insensitive
        assert exact_match(MockExample(), MockPrediction()) is True
        
        MockPrediction.response = "world"
        assert exact_match(MockExample(), MockPrediction()) is False
    
    def test_contains_answer(self):
        from fair_prompt_optimizer.metrics import contains_answer
        
        class MockExample:
            response = "42"
        
        class MockPrediction:
            response = "The answer is 42."
        
        assert contains_answer(MockExample(), MockPrediction()) is True
        
        MockPrediction.response = "The answer is 43."
        assert contains_answer(MockExample(), MockPrediction()) is False
    
    def test_numeric_accuracy(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        class MockExample:
            response = "42.0"
        
        class MockPrediction:
            response = "The result is 42."
        
        assert numeric_accuracy(MockExample(), MockPrediction()) is True
        
        MockPrediction.response = "41.5"  # Within 1% tolerance
        assert numeric_accuracy(MockExample(), MockPrediction()) is True
        
        MockPrediction.response = "50"  # Outside tolerance
        assert numeric_accuracy(MockExample(), MockPrediction()) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
