# tests/test_cli.py
"""
Unit tests for fair_prompt_optimizer CLI module.

Run with: pytest tests/test_cli.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestCLIInit:
    """Test CLI init command."""

    def test_init_simple_llm_template(self):
        """Test generating simple_llm template."""
        from fair_prompt_optimizer.cli import cmd_init

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.json"

            args = Mock()
            args.type = "simple_llm"
            args.output = str(output_path)

            cmd_init(args)

            assert output_path.exists()
            with open(output_path) as f:
                config = json.load(f)

            assert config["type"] == "simple_llm"
            assert "prompts" in config
            assert "model" in config

    def test_init_agent_template(self):
        """Test generating agent template."""
        from fair_prompt_optimizer.cli import cmd_init

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "agent.json"

            args = Mock()
            args.type = "agent"
            args.output = str(output_path)

            cmd_init(args)

            assert output_path.exists()
            with open(output_path) as f:
                config = json.load(f)

            assert config["type"] == "agent"
            assert "agent" in config
            assert "planner_type" in config["agent"]

    def test_init_multi_agent_template(self):
        """Test generating multi_agent template."""
        from fair_prompt_optimizer.cli import cmd_init

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "multi.json"

            args = Mock()
            args.type = "multi_agent"
            args.output = str(output_path)

            cmd_init(args)

            assert output_path.exists()
            with open(output_path) as f:
                config = json.load(f)

            assert config["type"] == "multi_agent"
            assert "manager" in config
            assert "workers" in config


class TestCLIInfo:
    """Test CLI info command."""

    def test_info_simple_llm(self):
        """Test info command on simple_llm config."""
        from fair_prompt_optimizer.cli import cmd_info

        config = {
            "version": "1.0",
            "type": "simple_llm",
            "prompts": {
                "system_prompt": "You are a classifier.",
                "examples": ["ex1", "ex2"],
            },
            "model": {
                "adapter": "TestAdapter",
                "model_name": "test-model",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            args = Mock()
            args.config = str(config_path)
            args.verbose = False

            # Should not raise
            cmd_info(args)

    def test_info_with_optimization(self):
        """Test info command shows optimization details."""
        from fair_prompt_optimizer.cli import cmd_info

        config = {
            "version": "1.0",
            "type": "agent",
            "prompts": {"role_definition": "Test", "examples": []},
            "model": {"adapter": "Test", "model_name": "test"},
            "agent": {"planner_type": "Test", "tools": []},
            "optimization": {
                "runs": [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "optimizer": "bootstrap",
                        "metric": "accuracy",
                        "examples_before": 0,
                        "examples_after": 3,
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            args = Mock()
            args.config = str(config_path)
            args.verbose = False

            # Should not raise
            cmd_info(args)


class TestCLITest:
    """Test CLI test command."""

    def test_test_simple_llm(self):
        """Test test command on simple_llm config."""
        from fair_prompt_optimizer.cli import cmd_test

        config = {
            "version": "1.0",
            "type": "simple_llm",
            "prompts": {"system_prompt": "Classify sentiment."},
            "model": {"adapter": "MockAdapter", "model_name": "mock"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            args = Mock()
            args.config = str(config_path)
            args.input = "Test input"
            args.model = None
            args.adapter = None

            # Mock create_llm to return a mock LLM
            mock_response = Mock()
            mock_response.content = "SENTIMENT: positive"

            mock_llm = Mock()
            mock_llm.invoke = Mock(return_value=mock_response)

            with patch("fair_prompt_optimizer.cli.create_llm", return_value=mock_llm):
                cmd_test(args)


class TestCLIOptimize:
    """Test CLI optimize command."""

    def test_optimize_missing_config_raises(self):
        """Test optimize with missing config file."""
        from fair_prompt_optimizer.cli import cmd_optimize

        args = Mock()
        args.config = "/nonexistent/path/config.json"
        args.training = "/some/training.json"
        args.output = None
        args.metric = "contains_answer"
        args.optimizer = "bootstrap"
        args.quiet = False
        args.dry_run = False

        # New validation logic exits with sys.exit(1)
        with pytest.raises(SystemExit) as excinfo:
            cmd_optimize(args)
        assert excinfo.value.code == 1

    def test_optimize_missing_training_raises(self):
        """Test optimize with missing training file."""
        from fair_prompt_optimizer.cli import cmd_optimize

        # Create a valid config to pass config validation
        config = {
            "version": "1.0",
            "type": "simple_llm",
            "prompts": {"role_definition": "Test prompt"},
            "model": {"adapter": "HuggingFaceAdapter", "model_name": "test-model"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            args = Mock()
            args.config = str(config_path)
            args.training = "/nonexistent/training.json"
            args.output = None
            args.metric = "contains_answer"
            args.optimizer = "bootstrap"
            args.quiet = False
            args.dry_run = False

            # New validation logic exits with sys.exit(1)
            with pytest.raises(SystemExit) as excinfo:
                cmd_optimize(args)
            assert excinfo.value.code == 1


class TestCLICreateLLM:
    """Test create_llm helper function."""

    def test_create_llm_huggingface(self):
        """Test creating HuggingFace adapter."""
        from fair_prompt_optimizer.cli import create_llm

        model_config = {
            "adapter": "HuggingFaceAdapter",
            "model_name": "test-model",
            "adapter_kwargs": {},
        }

        # Mock the HuggingFaceAdapter at the fairlib module level
        with patch("fairlib.HuggingFaceAdapter") as mock_adapter:
            mock_adapter.return_value = Mock()
            create_llm(model_config)
            mock_adapter.assert_called_once_with("test-model")

    def test_create_llm_with_override(self):
        """Test creating LLM with model override."""
        from fair_prompt_optimizer.cli import create_llm

        model_config = {
            "adapter": "HuggingFaceAdapter",
            "model_name": "original-model",
        }

        with patch("fairlib.HuggingFaceAdapter") as mock_adapter:
            mock_adapter.return_value = Mock()
            create_llm(model_config, model_override="override-model")
            mock_adapter.assert_called_once_with("override-model")

    def test_create_llm_unknown_adapter(self):
        """Test creating LLM with unknown adapter raises."""
        from fair_prompt_optimizer.cli import create_llm

        model_config = {
            "adapter": "UnknownAdapter",
            "model_name": "test",
        }

        with pytest.raises(ValueError, match="Unknown adapter"):
            create_llm(model_config)


class TestCLIMetricResolution:
    """Test metric resolution via getattr on metrics module."""

    def test_builtin_metrics_accessible(self):
        """Test that built-in metrics are accessible via getattr."""
        from fair_prompt_optimizer import metrics as metrics_module
        from fair_prompt_optimizer.metrics import (
            contains_answer,
            exact_match,
            fuzzy_match,
            numeric_accuracy,
        )

        assert getattr(metrics_module, "exact_match") == exact_match
        assert getattr(metrics_module, "contains_answer") == contains_answer
        assert getattr(metrics_module, "numeric_accuracy") == numeric_accuracy
        assert getattr(metrics_module, "fuzzy_match") == fuzzy_match

    def test_unknown_metric_raises(self):
        """Test getting unknown metric raises error."""
        from fair_prompt_optimizer import metrics as metrics_module

        with pytest.raises(AttributeError):
            getattr(metrics_module, "nonexistent_metric")


class TestCLIMainParser:
    """Test main CLI argument parser."""

    def test_help_does_not_crash(self):
        """Test that --help doesn't crash."""
        from fair_prompt_optimizer.cli import main

        with pytest.raises(SystemExit) as excinfo:
            with patch.object(sys, "argv", ["fair-optimize", "--help"]):
                main()

        # --help exits with code 0
        assert excinfo.value.code == 0

    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        from fair_prompt_optimizer.cli import main

        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["fair-optimize"]):
                main()


class TestCLICompare:
    """Test CLI compare command."""

    def test_compare_two_configs(self):
        """Test comparing two config files."""
        from fair_prompt_optimizer.cli import cmd_compare

        config1 = {
            "version": "1.0",
            "type": "agent",
            "prompts": {"role_definition": "Role 1", "examples": ["ex1"]},
            "model": {"adapter": "Test", "model_name": "test"},
        }

        config2 = {
            "version": "1.0",
            "type": "agent",
            "prompts": {"role_definition": "Role 2", "examples": ["ex1", "ex2", "ex3"]},
            "model": {"adapter": "Test", "model_name": "test"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "config1.json"
            path2 = Path(tmpdir) / "config2.json"

            with open(path1, "w") as f:
                json.dump(config1, f)
            with open(path2, "w") as f:
                json.dump(config2, f)

            args = Mock()
            args.config1 = str(path1)
            args.config2 = str(path2)

            # Should not raise
            cmd_compare(args)


class TestCLIValidate:
    """Test CLI validate command and validation functions."""

    def test_validate_config_valid(self):
        """Test validating a valid config."""
        from fair_prompt_optimizer.cli import validate_config

        config = {
            "version": "1.0",
            "type": "agent",
            "prompts": {"role_definition": "You are a helpful assistant."},
            "model": {"adapter": "HuggingFaceAdapter", "model_name": "test-model"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            is_valid, errors, warnings = validate_config(str(config_path))

            assert is_valid
            assert len(errors) == 0

    def test_validate_config_missing_model(self):
        """Test validating config with missing model."""
        from fair_prompt_optimizer.cli import validate_config

        config = {
            "type": "agent",
            "prompts": {"role_definition": "Test"},
            "model": {},  # Empty model
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            is_valid, errors, warnings = validate_config(str(config_path))

            assert not is_valid
            assert any("model" in e.lower() for e in errors)

    def test_validate_config_placeholder_warning(self):
        """Test that placeholder content generates warnings."""
        from fair_prompt_optimizer.cli import validate_config

        config = {
            "type": "agent",
            "prompts": {"role_definition": "# TODO: Define your role"},
            "model": {"adapter": "HuggingFaceAdapter", "model_name": "test-model"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            is_valid, errors, warnings = validate_config(str(config_path))

            assert is_valid  # Warnings don't fail validation
            assert any("placeholder" in w.lower() for w in warnings)

    def test_validate_training_valid(self):
        """Test validating valid training data."""
        from fair_prompt_optimizer.cli import validate_training_examples

        examples = [
            {"inputs": {"user_input": "Hello"}, "expected_output": "Hi"},
            {"inputs": {"user_input": "Bye"}, "expected_output": "Goodbye"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            training_path = Path(tmpdir) / "training.json"
            with open(training_path, "w") as f:
                json.dump(examples, f)

            is_valid, errors, warnings = validate_training_examples(str(training_path))

            assert is_valid
            assert len(errors) == 0

    def test_validate_training_missing_fields(self):
        """Test validating training data with missing fields."""
        from fair_prompt_optimizer.cli import validate_training_examples

        examples = [
            {"inputs": {}, "expected_output": "Hi"},  # Missing user_input
            {"inputs": {"user_input": "Test"}},  # Missing expected_output
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            training_path = Path(tmpdir) / "training.json"
            with open(training_path, "w") as f:
                json.dump(examples, f)

            is_valid, errors, warnings = validate_training_examples(str(training_path))

            assert not is_valid
            assert len(errors) >= 2

    def test_validate_training_few_examples_warning(self):
        """Test that few examples generates a warning."""
        from fair_prompt_optimizer.cli import validate_training_examples

        examples = [
            {"inputs": {"user_input": "Test"}, "expected_output": "Result"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            training_path = Path(tmpdir) / "training.json"
            with open(training_path, "w") as f:
                json.dump(examples, f)

            is_valid, errors, warnings = validate_training_examples(str(training_path))

            assert is_valid
            assert any("recommend" in w.lower() or "only" in w.lower() for w in warnings)

    def test_validate_command(self):
        """Test the validate command."""
        from fair_prompt_optimizer.cli import cmd_validate

        config = {
            "type": "agent",
            "prompts": {"role_definition": "Test"},
            "model": {"adapter": "HuggingFaceAdapter", "model_name": "test-model"},
        }

        examples = [
            {"inputs": {"user_input": "Test"}, "expected_output": "Result"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            training_path = Path(tmpdir) / "training.json"

            with open(config_path, "w") as f:
                json.dump(config, f)
            with open(training_path, "w") as f:
                json.dump(examples, f)

            args = Mock()
            args.config = str(config_path)
            args.training = str(training_path)

            # Should not raise
            cmd_validate(args)


class TestCLIColors:
    """Test CLI color output helpers."""

    def test_colors_class_has_codes(self):
        """Test that Colors class has color codes."""
        from fair_prompt_optimizer.cli import Colors

        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "RESET")

    def test_print_helpers_dont_crash(self):
        """Test that print helpers work without crashing."""
        from fair_prompt_optimizer.cli import (
            print_error,
            print_info,
            print_step,
            print_success,
            print_warning,
        )

        # These should not raise
        print_error("Test error", "Test suggestion")
        print_warning("Test warning")
        print_success("Test success")
        print_info("Test info")
        print_step(1, 3, "Test step")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
