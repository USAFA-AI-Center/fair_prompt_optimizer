# tests/test_config.py
"""
Unit tests for fair_prompt_optimizer config module.

Run with: pytest tests/test_config.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestPromptsData:
    """Test PromptsData serialization."""
    
    def test_to_dict_empty(self):
        from fair_prompt_optimizer.config import OptimizedConfig
        
        config = OptimizedConfig()
        data = config.to_dict()
        
        assert "prompts" not in data or data.get("prompts") == {}
        assert "optimization" in data
    
    def test_to_dict_with_prompts(self):
        from fair_prompt_optimizer.config import OptimizedConfig
        
        config = OptimizedConfig(config={
            "version": "1.0",
            "type": "agent",
            "prompts": {
                "role_definition": "You are a helper.",
                "tool_instructions": [{"name": "calc", "description": "Does math"}],
                "format_instructions": ["Be concise"],
                "examples": ["Example 1"],
            }
        })
        
        data = config.to_dict()
        
        assert data["prompts"]["role_definition"] == "You are a helper."
        assert len(data["prompts"]["tool_instructions"]) == 1
        assert len(data["prompts"]["examples"]) == 1
    
    def test_round_trip(self):
        from fair_prompt_optimizer.config import OptimizedConfig
        
        original = {
            "version": "1.0",
            "type": "agent",
            "prompts": {
                "role_definition": "Test role",
                "tool_instructions": [{"name": "tool1", "description": "desc1"}],
                "worker_instructions": [],
                "format_instructions": ["Format 1", "Format 2"],
                "examples": ["Ex 1", "Ex 2", "Ex 3"],
            },
            "model": {"adapter": "TestAdapter", "model_name": "test-model"},
            "agent": {"planner_type": "TestPlanner", "tools": [], "max_steps": 5},
        }
        
        config = OptimizedConfig.from_dict(original)
        restored = config.to_dict()
        
        # Remove optimization for comparison (it's added)
        del restored["optimization"]
        
        assert restored == original


class TestOptimizationProvenance:
    """Test optimization provenance tracking."""
    
    def test_record_run(self):
        from fair_prompt_optimizer.config import OptimizationProvenance
        
        prov = OptimizationProvenance()
        assert prov.optimized == False
        assert len(prov.runs) == 0
        
        prov.record_run(
            optimizer="bootstrap",
            metric="accuracy",
            num_examples=10,
            examples_before=0,
            examples_after=3,
        )
        
        assert prov.optimized == True
        assert prov.optimizer == "bootstrap"
        assert prov.metric == "accuracy"
        assert len(prov.runs) == 1
        assert prov.runs[0].examples_before == 0
        assert prov.runs[0].examples_after == 3
    
    def test_multiple_runs(self):
        from fair_prompt_optimizer.config import OptimizationProvenance
        
        prov = OptimizationProvenance()
        
        prov.record_run(optimizer="bootstrap", metric="accuracy", num_examples=10)
        prov.record_run(optimizer="mipro", metric="f1", num_examples=20)
        
        assert len(prov.runs) == 2
        assert prov.optimizer == "mipro"  # Most recent
        assert prov.metric == "f1"
    
    def test_serialization(self):
        from fair_prompt_optimizer.config import OptimizationProvenance
        
        prov = OptimizationProvenance()
        prov.record_run(optimizer="bootstrap", metric="accuracy", num_examples=10)
        
        data = prov.to_dict()
        restored = OptimizationProvenance.from_dict(data)
        
        assert restored.optimized == prov.optimized
        assert restored.optimizer == prov.optimizer
        assert len(restored.runs) == len(prov.runs)


class TestConfigIO:
    """Test config file I/O."""
    
    def test_save_and_load(self):
        from fair_prompt_optimizer.config import (
            OptimizedConfig,
            save_optimized_config,
            load_optimized_config,
        )
        
        config = OptimizedConfig(config={
            "version": "1.0",
            "type": "agent",
            "prompts": {
                "role_definition": "Test",
                "examples": ["ex1", "ex2"],
            }
        })
        config.optimization.record_run(
            optimizer="bootstrap",
            metric="test_metric",
            num_examples=5,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"
            save_optimized_config(config, str(path))
            
            loaded = load_optimized_config(str(path))
            
            assert loaded.prompts["role_definition"] == "Test"
            assert len(loaded.prompts["examples"]) == 2
            assert loaded.optimization.optimized == True
            assert loaded.optimization.optimizer == "bootstrap"
    
    def test_load_fairlib_config(self):
        """Test loading a config without optimization section (from fairlib)."""
        from fair_prompt_optimizer.config import load_optimized_config
        
        fairlib_config = {
            "version": "1.0",
            "type": "agent",
            "prompts": {
                "role_definition": "From fairlib",
                "examples": [],
            },
            "model": {"adapter": "Test", "model_name": "test"},
            "agent": {"planner_type": "Test", "tools": []},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fairlib_config.json"
            with open(path, 'w') as f:
                json.dump(fairlib_config, f)
            
            loaded = load_optimized_config(str(path))
            
            assert loaded.prompts["role_definition"] == "From fairlib"
            assert loaded.optimization.optimized == False  # No optimization yet


class TestTrainingExamples:
    """Test training examples I/O."""
    
    def test_load_examples(self):
        from fair_prompt_optimizer.config import load_training_examples, TrainingExample
        
        examples_data = [
            {"inputs": {"user_input": "Hello"}, "expected_output": "Hi"},
            {"inputs": {"user_input": "2+2"}, "expected_output": "4"},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "examples.json"
            with open(path, 'w') as f:
                json.dump(examples_data, f)
            
            examples = load_training_examples(str(path))
            
            assert len(examples) == 2
            assert examples[0].inputs["user_input"] == "Hello"
            assert examples[1].expected_output == "4"
    
    def test_dspy_translation(self):
        from fair_prompt_optimizer.config import TrainingExample, DSPyTranslator

        examples = [
            TrainingExample(inputs={"user_input": "Test"}, expected_output="Result"),
        ]

        translator = DSPyTranslator()
        dspy_examples = translator.to_dspy_examples(examples)

        assert len(dspy_examples) == 1
        assert dspy_examples[0].user_input == "Test"
        # Output field defaults to "expected_output"
        assert dspy_examples[0].expected_output == "Result"

    def test_dspy_translation_custom_fields(self):
        from fair_prompt_optimizer.config import TrainingExample, DSPyTranslator

        examples = [
            TrainingExample(inputs={"query": "Test query"}, expected_output="Result"),
        ]

        translator = DSPyTranslator(input_field="query", output_field="answer")
        dspy_examples = translator.to_dspy_examples(examples)

        assert len(dspy_examples) == 1
        assert dspy_examples[0].query == "Test query"
        assert dspy_examples[0].answer == "Result"


class TestFileHash:
    """Test file hashing for provenance."""
    
    def test_compute_hash(self):
        from fair_prompt_optimizer.config import compute_file_hash
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, 'w') as f:
                f.write('{"test": "data"}')
            
            hash1 = compute_file_hash(str(path))
            hash2 = compute_file_hash(str(path))
            
            assert hash1 == hash2
            assert hash1.startswith("sha256:")
    
    def test_hash_changes_with_content(self):
        from fair_prompt_optimizer.config import compute_file_hash
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            
            with open(path, 'w') as f:
                f.write('{"version": 1}')
            hash1 = compute_file_hash(str(path))
            
            with open(path, 'w') as f:
                f.write('{"version": 2}')
            hash2 = compute_file_hash(str(path))
            
            assert hash1 != hash2


class TestTrainingExampleFullTrace:
    """Test TrainingExample with full_trace field."""

    def test_full_trace_attribute(self):
        from fair_prompt_optimizer.config import TrainingExample

        example = TrainingExample(
            inputs={"user_input": "What is 2+2?"},
            expected_output="4",
            full_trace="User: What is 2+2?\nThought: Calculate\nAction: calc(2+2)\nAnswer: 4",
        )

        assert example.full_trace is not None
        assert "Thought" in example.full_trace
        assert "Action" in example.full_trace

    def test_full_trace_none_by_default(self):
        from fair_prompt_optimizer.config import TrainingExample

        example = TrainingExample(
            inputs={"user_input": "test"},
            expected_output="result",
        )

        assert example.full_trace is None


class TestOptimizedConfigAccessors:
    """Test OptimizedConfig property accessors."""

    def test_format_instructions_accessor(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(
            config={
                "prompts": {"format_instructions": ["Format 1", "Format 2"]}
            }
        )

        assert config.format_instructions == ["Format 1", "Format 2"]

    def test_format_instructions_setter(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(config={"prompts": {}})
        config.format_instructions = ["New format"]

        assert config.format_instructions == ["New format"]

    def test_type_accessor(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(config={"type": "agent"})
        assert config.type == "agent"

    def test_type_accessor_default(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(config={})
        assert config.type == "agent"  # Default value


class TestOptimizationRunDetails:
    """Test OptimizationRun with detailed config."""

    def test_run_with_optimizer_config(self):
        from fair_prompt_optimizer.config import OptimizationProvenance

        prov = OptimizationProvenance()
        prov.record_run(
            optimizer="mipro",
            metric="accuracy",
            num_examples=20,
            examples_before=0,
            examples_after=5,
            role_definition_changed=True,
            format_instructions_changed=True,
            optimizer_config={
                "max_bootstrapped_demos": 4,
                "mipro_auto": "medium",
            },
        )

        run = prov.runs[0]
        assert run.role_definition_changed == True
        assert run.format_instructions_changed == True
        assert run.optimizer_config["max_bootstrapped_demos"] == 4
        assert run.optimizer_config["mipro_auto"] == "medium"


class TestSaveTrainingExamples:
    """Test saving training examples."""

    def test_save_and_reload(self):
        from fair_prompt_optimizer.config import (
            TrainingExample,
            save_training_examples,
            load_training_examples,
        )

        examples = [
            TrainingExample(
                inputs={"user_input": "Hello"},
                expected_output="Hi",
                full_trace="User: Hello\nAssistant: Hi",
            ),
            TrainingExample(
                inputs={"user_input": "Bye"},
                expected_output="Goodbye",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "examples.json"
            save_training_examples(examples, str(path))

            loaded = load_training_examples(str(path))

            assert len(loaded) == 2
            assert loaded[0].inputs["user_input"] == "Hello"
            assert loaded[0].full_trace is not None
            assert loaded[1].full_trace is None


class TestOptimizedConfigFromDict:
    """Test OptimizedConfig.from_dict edge cases."""

    def test_from_dict_with_nested_optimization(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        data = {
            "version": "1.0",
            "type": "agent",
            "prompts": {"examples": []},
            "optimization": {
                "runs": [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "optimizer": "bootstrap",
                        "metric": "test",
                        "examples_before": 0,
                        "examples_after": 3,
                        "optimizer_config": {"key": "value"},
                    }
                ]
            },
        }

        config = OptimizedConfig.from_dict(data)

        assert config.optimization.optimized == True
        assert len(config.optimization.runs) == 1
        assert config.optimization.runs[0].optimizer == "bootstrap"

    def test_from_dict_minimal(self):
        from fair_prompt_optimizer.config import OptimizedConfig

        data = {"prompts": {}}
        config = OptimizedConfig.from_dict(data)

        assert config.optimization.optimized == False


class TestComputeFileHashEdgeCases:
    """Test compute_file_hash edge cases."""

    def test_hash_empty_file(self):
        from fair_prompt_optimizer.config import compute_file_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.touch()

            hash_val = compute_file_hash(str(path))
            assert hash_val.startswith("sha256:")

    def test_hash_large_file(self):
        from fair_prompt_optimizer.config import compute_file_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.json"
            # Create a file with substantial content
            with open(path, "w") as f:
                json.dump({"data": "x" * 100000}, f)

            hash_val = compute_file_hash(str(path))
            assert hash_val.startswith("sha256:")
            assert len(hash_val) > 10  # Has actual hash content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])