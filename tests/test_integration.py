# tests/test_integration.py
"""
Integration tests for fair_prompt_optimizer with fairlib.

These tests require fairlib to be installed.

Run with: pytest tests/test_integration.py -v
"""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if fairlib not available
fairlib_available = False
try:
    from fairlib.core.prompts import PromptBuilder, RoleDefinition, ToolInstruction
    from fairlib.utils.config_manager import apply_prompts, extract_prompts

    fairlib_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not fairlib_available, reason="fairlib not installed")


class TestPromptBuilderIntegration:
    """Test integration with fairlib's PromptBuilder."""

    def test_extract_prompts(self):
        """Test extracting prompts from a PromptBuilder."""
        from fairlib.core.prompts import (
            Example,
            FormatInstruction,
            PromptBuilder,
            RoleDefinition,
            ToolInstruction,
        )
        from fairlib.utils.config_manager import extract_prompts

        builder = PromptBuilder()
        builder.role_definition = RoleDefinition("You are a helpful assistant.")
        builder.tool_instructions.append(ToolInstruction("calc", "Does math"))
        builder.format_instructions.append(FormatInstruction("Be concise"))
        builder.examples.append(Example("Example 1"))

        prompts = extract_prompts(builder)

        assert prompts["role_definition"] == "You are a helpful assistant."
        assert len(prompts["tool_instructions"]) == 1
        assert prompts["tool_instructions"][0]["name"] == "calc"
        assert len(prompts["format_instructions"]) == 1
        assert len(prompts["examples"]) == 1

    def test_apply_prompts(self):
        """Test applying prompts to a PromptBuilder."""
        from fairlib.core.prompts import PromptBuilder
        from fairlib.utils.config_manager import apply_prompts

        prompts = {
            "role_definition": "Test role",
            "tool_instructions": [{"name": "tool1", "description": "desc1"}],
            "worker_instructions": [],
            "format_instructions": ["Format 1"],
            "examples": ["Ex 1", "Ex 2"],
        }

        builder = PromptBuilder()
        apply_prompts(prompts, builder)

        assert builder.role_definition.text == "Test role"
        assert len(builder.tool_instructions) == 1
        assert builder.tool_instructions[0].name == "tool1"
        assert len(builder.format_instructions) == 1
        assert len(builder.examples) == 2

    def test_round_trip(self):
        """Test extract -> modify -> apply round trip."""
        from fairlib.core.prompts import Example, PromptBuilder, RoleDefinition
        from fairlib.utils.config_manager import apply_prompts, extract_prompts

        # Create original
        original = PromptBuilder()
        original.role_definition = RoleDefinition("Original role")
        original.examples.append(Example("Original example"))

        # Extract
        prompts = extract_prompts(original)

        # Modify (simulating optimization)
        prompts["examples"].append("New optimized example")

        # Apply to new builder
        restored = PromptBuilder()
        apply_prompts(prompts, restored)

        assert restored.role_definition.text == "Original role"
        assert len(restored.examples) == 2
        assert restored.examples[1].text == "New optimized example"


class TestConfigWithFairlib:
    """Test OptimizedConfig integration with fairlib."""

    def test_get_prompt_builder(self):
        """Test creating PromptBuilder from config."""
        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(
            config={
                "version": "1.0",
                "type": "agent",
                "prompts": {
                    "role_definition": "Test role",
                    "tool_instructions": [{"name": "calc", "description": "math"}],
                    "format_instructions": ["Be clear"],
                    "examples": ["Example 1"],
                },
            }
        )

        builder = config.get_prompt_builder()

        assert builder.role_definition.text == "Test role"
        assert len(builder.tool_instructions) == 1
        assert len(builder.examples) == 1

    def test_update_from_prompt_builder(self):
        """Test updating config from PromptBuilder."""
        from fairlib.core.prompts import Example, PromptBuilder, RoleDefinition

        from fair_prompt_optimizer.config import OptimizedConfig

        config = OptimizedConfig(
            config={
                "version": "1.0",
                "type": "agent",
                "prompts": {"role_definition": "Old", "examples": []},
            }
        )

        # Create updated builder
        builder = PromptBuilder()
        builder.role_definition = RoleDefinition("New role")
        builder.examples.append(Example("New example"))

        # Update config
        config.update_from_prompt_builder(builder)

        assert config.prompts["role_definition"] == "New role"
        assert len(config.prompts["examples"]) == 1


class TestSaveLoadWithFairlib:
    """Test that configs saved by optimizer load in fairlib."""

    def test_optimized_config_loadable_by_fairlib(self):
        """Test that an optimized config can be loaded by fairlib."""
        from fairlib.utils.config_manager import load_agent_config

        from fair_prompt_optimizer.config import OptimizedConfig, save_optimized_config

        # Create optimized config
        config = OptimizedConfig(
            config={
                "version": "1.0",
                "type": "agent",
                "prompts": {
                    "role_definition": "Optimized role",
                    "tool_instructions": [],
                    "format_instructions": [],
                    "examples": ["Optimized example 1", "Optimized example 2"],
                },
                "model": {"adapter": "TestAdapter", "model_name": "test"},
                "agent": {"planner_type": "TestPlanner", "tools": [], "max_steps": 5},
            }
        )

        # Add optimization provenance
        config.optimization.record_run(
            optimizer="bootstrap",
            metric="accuracy",
            num_examples=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimized.json"
            save_optimized_config(config, str(path))

            # Load with fairlib's loader
            fairlib_config = load_agent_config(str(path))

            # Verify fairlib can read it
            assert fairlib_config["prompts"]["role_definition"] == "Optimized role"
            assert len(fairlib_config["prompts"]["examples"]) == 2

            # Optimization section is present (fairlib ignores it)
            assert "optimization" in fairlib_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
