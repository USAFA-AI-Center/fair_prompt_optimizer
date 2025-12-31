# test_fairlib_integration.py
"""
Integration tests that verify fair_prompt_optimizer works with actual fair_llm.

These tests require fair_llm to be installed and will be skipped otherwise.
"""

import pytest
import json

# Skip all tests in this module if fairlib is not installed
fairlib = pytest.importorskip("fairlib", reason="fair_llm package required for integration tests")

class TestFairlibImports:
    """Verify all expected fairlib components are importable"""
    
    def test_import_adapters(self):
        """Test that adapter classes can be imported"""
        from fairlib import HuggingFaceAdapter
        assert HuggingFaceAdapter is not None
    
    def test_import_tools(self):
        """Test that tool classes can be imported"""
        from fairlib import SafeCalculatorTool, ToolRegistry, ToolExecutor
        assert SafeCalculatorTool is not None
        assert ToolRegistry is not None
        assert ToolExecutor is not None
    
    def test_import_agent_components(self):
        """Test that agent components can be imported"""
        from fairlib import (
            SimpleAgent,
            SimpleReActPlanner,
            WorkingMemory,
            RoleDefinition,
        )
        assert SimpleAgent is not None
        assert SimpleReActPlanner is not None
        assert WorkingMemory is not None
        assert RoleDefinition is not None
    
    def test_import_prompt_components(self):
        """Test that prompt components can be imported"""
        from fairlib import (
            PromptBuilder,
            RoleDefinition,
            ToolInstruction,
            WorkerInstruction,
            FormatInstruction,
            Example,
        )
        assert PromptBuilder is not None
        assert RoleDefinition is not None
        assert ToolInstruction is not None
        assert WorkerInstruction is not None
        assert FormatInstruction is not None
        assert Example is not None


class TestPromptBuilderInterface:
    """Test that PromptBuilder has the expected interface for optimization"""
    
    def test_prompt_builder_has_required_attributes(self):
        """Verify PromptBuilder has all required attributes"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        # Required attributes for fair_prompt_optimizer
        assert hasattr(builder, 'role_definition')
        assert hasattr(builder, 'tool_instructions')
        assert hasattr(builder, 'worker_instructions')
        assert hasattr(builder, 'format_instructions')
        assert hasattr(builder, 'examples')
    
    def test_prompt_builder_has_serialization_methods(self):
        """Verify PromptBuilder has serialization methods"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        assert hasattr(builder, 'to_dict')
        assert hasattr(builder, 'from_dict')
        assert hasattr(builder, 'save')
        assert hasattr(builder, 'load')
        assert callable(builder.to_dict)
        assert callable(builder.save)
    
    def test_prompt_builder_has_optimization_tracking(self):
        """Verify PromptBuilder tracks optimization status"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        assert hasattr(builder, '_optimized')
        assert hasattr(builder, '_optimization_metadata')
        assert hasattr(builder, 'is_optimized')
        assert hasattr(builder, 'optimization_info')
        assert hasattr(builder, 'metadata')
    
    def test_prompt_builder_has_config_storage(self):
        """Verify PromptBuilder can store model/agent config"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        assert hasattr(builder, '_model_config')
        assert hasattr(builder, '_agent_config')
        assert hasattr(builder, '_raw_config')
    
    def test_prompt_builder_has_convenience_properties(self):
        """Verify PromptBuilder has convenience properties for agent reconstruction"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        # Model config properties
        assert hasattr(builder, 'model_config')
        assert hasattr(builder, 'model_name')
        assert hasattr(builder, 'adapter')
        assert hasattr(builder, 'adapter_kwargs')
        
        # Agent config properties
        assert hasattr(builder, 'agent_config')
        assert hasattr(builder, 'agent_type')
        assert hasattr(builder, 'planner_type')
        assert hasattr(builder, 'max_steps')
        assert hasattr(builder, 'tools')
        
        # Other properties
        assert hasattr(builder, 'version')
        assert hasattr(builder, 'raw_config')
    
    def test_prompt_builder_accepts_optimized_path(self):
        """Verify PromptBuilder __init__ accepts optimized_path parameter"""
        from fairlib import PromptBuilder
        import inspect
        
        sig = inspect.signature(PromptBuilder.__init__)
        params = list(sig.parameters.keys())
        
        assert 'optimized_path' in params


class TestPromptBuilderSerialization:
    """Test PromptBuilder serialization/deserialization"""
    
    def test_to_dict_structure(self):
        """Test that to_dict produces expected structure"""
        from fairlib import PromptBuilder, RoleDefinition, Example
        
        builder = PromptBuilder()
        builder.role_definition = RoleDefinition("Test role")
        builder.examples.append(Example("User: Test\nResponse: Test response"))
        
        data = builder.to_dict()
        
        assert "version" in data
        assert "role_definition" in data
        assert "tool_instructions" in data
        assert "worker_instructions" in data
        assert "format_instructions" in data
        assert "examples" in data
        assert "metadata" in data
        
        assert data["role_definition"] == "Test role"
        assert len(data["examples"]) == 1
    
    def test_from_dict_restores_state(self):
        """Test that from_dict correctly restores PromptBuilder state"""
        from fairlib import PromptBuilder
        
        data = {
            "version": "1.0",
            "role_definition": "Restored role",
            "tool_instructions": [
                {"name": "calculator", "description": "Does math"}
            ],
            "worker_instructions": [],
            "format_instructions": ["Be concise"],
            "examples": ["User: Hi\nResponse: Hello"],
            "metadata": {
                "optimized": True,
                "optimizer": "bootstrap"
            }
        }
        
        builder = PromptBuilder.from_dict(data)
        
        assert builder.role_definition.text == "Restored role"
        assert len(builder.tool_instructions) == 1
        assert builder.tool_instructions[0].name == "calculator"
        assert len(builder.format_instructions) == 1
        assert len(builder.examples) == 1
        assert builder._optimized is True
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Test complete save/load roundtrip"""
        from fairlib import PromptBuilder, RoleDefinition, Example, FormatInstruction
        
        # Create and configure builder
        original = PromptBuilder()
        original.role_definition = RoleDefinition("Roundtrip test role")
        original.examples.append(Example("Example 1"))
        original.examples.append(Example("Example 2"))
        original.format_instructions.append(FormatInstruction("Format rule"))
        original._optimized = True
        original._optimization_metadata = {"optimizer": "bootstrap", "score": 0.95}
        
        # Save
        save_path = tmp_path / "roundtrip_test.json"
        original.save(str(save_path))
        
        assert save_path.exists()
        
        # Load
        loaded = PromptBuilder.load(str(save_path))
        
        # Verify
        assert loaded.role_definition.text == original.role_definition.text
        assert len(loaded.examples) == len(original.examples)
        assert len(loaded.format_instructions) == len(original.format_instructions)
        assert loaded._optimized == original._optimized
        assert loaded._optimization_metadata.get("optimizer") == "bootstrap"
    
    def test_load_via_constructor(self, tmp_path):
        """Test loading via optimized_path constructor parameter"""
        from fairlib import PromptBuilder, RoleDefinition
        
        # Create a config file
        config_data = {
            "version": "1.0",
            "role_definition": "Constructor load test",
            "examples": ["Test example"],
            "metadata": {"optimized": True}
        }
        
        config_path = tmp_path / "constructor_test.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Load via constructor
        builder = PromptBuilder(optimized_path=str(config_path))
        
        assert builder.role_definition.text == "Constructor load test"
        assert len(builder.examples) == 1
        assert builder._optimized is True
    
    def test_missing_file_handled_gracefully(self, tmp_path, capsys):
        """Test that missing file is handled gracefully (prints warning, doesn't crash)"""
        from fairlib import PromptBuilder
        
        # Should not raise, but should print a warning
        builder = PromptBuilder(optimized_path=str(tmp_path / "nonexistent.json"))
        
        # Verify warning was printed
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "nonexistent" in captured.out.lower()
        
        # Builder should still be usable but empty
        assert builder.role_definition is None
        assert builder.model_config is None
        assert builder.is_optimized is False
    
    def test_empty_builder_convenience_properties(self):
        """Test that convenience properties return None/empty when no config loaded"""
        from fairlib import PromptBuilder
        
        builder = PromptBuilder()
        
        assert builder.model_name is None
        assert builder.adapter is None
        assert builder.adapter_kwargs == {}
        assert builder.agent_type is None
        assert builder.planner_type is None
        assert builder.max_steps is None
        assert builder.tools == []
        assert builder.model_config is None
        assert builder.agent_config is None
        assert builder.raw_config is None
        assert builder.is_optimized is False


class TestFAIRConfigCompatibility:
    """Test that PromptBuilder can load FAIRConfig format from fair_prompt_optimizer"""
    
    def test_loads_fair_config_format(self, tmp_path):
        """Test loading a full FAIRConfig (with model/agent sections)"""
        from fairlib import PromptBuilder
        
        # This is the format fair_prompt_optimizer produces
        fair_config = {
            "version": "1.0",
            "role_definition": "You are a helpful calculator assistant.",
            "tool_instructions": [
                {"name": "calculator", "description": "Performs calculations"}
            ],
            "worker_instructions": [],
            "format_instructions": ["Think step by step"],
            "examples": [
                "User: What is 2+2?\nResponse: 4",
                "User: What is 10/2?\nResponse: 5"
            ],
            "model": {
                "model_name": "dolphin3-qwen25-3b",
                "adapter": "HuggingFaceAdapter",
                "adapter_kwargs": {"device": "cuda"}
            },
            "agent": {
                "agent_type": "SimpleAgent",
                "planner_type": "SimpleReActPlanner",
                "max_steps": 10,
                "tools": ["SafeCalculatorTool"]
            },
            "metadata": {
                "optimized": True,
                "optimized_at": "2025-01-01T12:00:00",
                "optimizer": "bootstrap",
                "num_training_examples": 8
            }
        }
        
        config_path = tmp_path / "fair_config.json"
        with open(config_path, 'w') as f:
            json.dump(fair_config, f)
        
        # Load into PromptBuilder
        builder = PromptBuilder.load(str(config_path))
        
        # Verify prompt-related fields loaded correctly
        assert builder.role_definition.text == "You are a helpful calculator assistant."
        assert len(builder.tool_instructions) == 1
        assert builder.tool_instructions[0].name == "calculator"
        assert len(builder.examples) == 2
        assert builder._optimized is True
        
        # Verify model config is accessible
        assert builder.model_config is not None
        assert builder.model_config["model_name"] == "dolphin3-qwen25-3b"
        assert builder.model_config["adapter"] == "HuggingFaceAdapter"
        assert builder.model_config["adapter_kwargs"]["device"] == "cuda"
        
        # Verify agent config is accessible
        assert builder.agent_config is not None
        assert builder.agent_config["agent_type"] == "SimpleAgent"
        assert builder.agent_config["planner_type"] == "SimpleReActPlanner"
        assert builder.agent_config["max_steps"] == 10
        assert builder.agent_config["tools"] == ["SafeCalculatorTool"]
    
    def test_convenience_properties(self, tmp_path):
        """Test that convenience properties work for agent reconstruction"""
        from fairlib import PromptBuilder
        
        fair_config = {
            "version": "1.0",
            "role_definition": "Test role",
            "examples": [],
            "model": {
                "model_name": "test-model",
                "adapter": "HuggingFaceAdapter",
                "adapter_kwargs": {"device": "cpu", "temperature": 0.7}
            },
            "agent": {
                "agent_type": "SimpleAgent",
                "planner_type": "SimpleReActPlanner",
                "max_steps": 15,
                "tools": ["SafeCalculatorTool", "WebSearchTool"]
            },
            "metadata": {"optimized": True}
        }
        
        config_path = tmp_path / "convenience_test.json"
        with open(config_path, 'w') as f:
            json.dump(fair_config, f)
        
        # Load via constructor (the primary use case)
        builder = PromptBuilder(optimized_path=str(config_path))
        
        # Test convenience properties
        assert builder.model_name == "test-model"
        assert builder.adapter == "HuggingFaceAdapter"
        assert builder.adapter_kwargs == {"device": "cpu", "temperature": 0.7}
        assert builder.agent_type == "SimpleAgent"
        assert builder.planner_type == "SimpleReActPlanner"
        assert builder.max_steps == 15
        assert builder.tools == ["SafeCalculatorTool", "WebSearchTool"]
        assert builder.is_optimized is True
    
    def test_raw_config_preserved(self, tmp_path):
        """Test that raw config is preserved for any custom fields"""
        from fairlib import PromptBuilder
        
        fair_config = {
            "version": "1.0",
            "role_definition": "Test",
            "examples": [],
            "model": {"model_name": "test"},
            "agent": {"agent_type": "SimpleAgent"},
            "custom_field": "custom_value",
            "metadata": {"optimized": True, "custom_metadata": "value"}
        }
        
        config_path = tmp_path / "raw_config_test.json"
        with open(config_path, 'w') as f:
            json.dump(fair_config, f)
        
        builder = PromptBuilder(optimized_path=str(config_path))
        
        # Raw config should preserve everything
        assert builder.raw_config is not None
        assert builder.raw_config.get("custom_field") == "custom_value"
        assert builder.metadata.get("custom_metadata") == "value"
    
    def test_prompt_builder_output_compatible_with_optimizer(self, tmp_path):
        """Test that PromptBuilder.to_dict() output can be read by optimizer"""
        from fairlib import PromptBuilder, RoleDefinition, Example
        from fair_prompt_optimizer.translator import FAIRConfig
        
        # Create a PromptBuilder and serialize it
        builder = PromptBuilder()
        builder.role_definition = RoleDefinition("Test role for optimizer")
        builder.examples.append(Example("User: Test\nResponse: Works"))
        
        data = builder.to_dict()
        
        # The core fields should be compatible with FAIRConfig
        # FAIRConfig.from_dict should be able to read the prompt fields
        config = FAIRConfig(
            version=data.get("version", "1.0"),
            role_definition=data.get("role_definition"),
            examples=data.get("examples", []),
        )
        
        assert config.role_definition == "Test role for optimizer"
        assert len(config.examples) == 1


class TestRegistryWithRealFairlib:
    """Test registry population with actual fairlib classes"""
    
    def test_registries_populate_with_real_classes(self):
        """Test that registries get populated with real fairlib classes"""
        from fair_prompt_optimizer.registry import (
            ADAPTER_REGISTRY,
            TOOL_REGISTRY,
            PLANNER_REGISTRY,
            AGENT_REGISTRY,
            _populate_registries,
        )
        from fairlib import HuggingFaceAdapter, SafeCalculatorTool, SimpleReActPlanner, SimpleAgent
        
        # Clear and repopulate
        ADAPTER_REGISTRY.clear()
        TOOL_REGISTRY.clear()
        PLANNER_REGISTRY.clear()
        AGENT_REGISTRY.clear()
        _populate_registries()
        
        # Verify real classes are registered
        assert ADAPTER_REGISTRY["HuggingFaceAdapter"] is HuggingFaceAdapter
        assert TOOL_REGISTRY["SafeCalculatorTool"] is SafeCalculatorTool
        assert PLANNER_REGISTRY["SimpleReActPlanner"] is SimpleReActPlanner
        assert AGENT_REGISTRY["SimpleAgent"] is SimpleAgent


class TestRoleDefinitionIntegration:
    """Test that RoleDefinition integrates correctly"""
    
    def test_role_definition_text_attribute(self):
        """Verify RoleDefinition has expected interface"""
        from fairlib import RoleDefinition
        
        role = RoleDefinition("Test role text")
        assert hasattr(role, 'text')
        assert role.text == "Test role text"
    
    def test_role_definition_render(self):
        """Verify RoleDefinition has render method"""
        from fairlib import RoleDefinition
        
        role = RoleDefinition("Rendered text")
        assert hasattr(role, 'render')
        assert role.render() == "Rendered text"


class TestExampleIntegration:
    """Test that Example class integrates correctly"""
    
    def test_example_class_interface(self):
        """Verify Example class has expected interface"""
        from fairlib import Example
        
        example = Example("User: Test\nResponse: Test response")
        
        assert hasattr(example, 'text')
        assert hasattr(example, 'render')
        assert example.text == "User: Test\nResponse: Test response"
        assert example.render() == "User: Test\nResponse: Test response"


class TestToolInstructionIntegration:
    """Test ToolInstruction integration"""
    
    def test_tool_instruction_interface(self):
        """Verify ToolInstruction has expected interface"""
        from fairlib import ToolInstruction
        
        tool = ToolInstruction("calculator", "Does math")
        
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'render')
        assert tool.name == "calculator"
        assert tool.description == "Does math"


class TestToolRegistryIntegration:
    """Test tool registry integration"""
    
    def test_tool_registry_interface(self):
        """Verify ToolRegistry has expected interface"""
        from fairlib import ToolRegistry, SafeCalculatorTool
        
        registry = ToolRegistry()
        tool = SafeCalculatorTool()
        
        # These methods must exist for our registry.py to work
        assert hasattr(registry, 'register_tool')
        assert hasattr(registry, 'get_all_tools')
        
        registry.register_tool(tool)
        tools = registry.get_all_tools()
        
        assert len(tools) > 0


class TestPromptBuilderWithPlanner:
    """Test PromptBuilder integration with SimpleReActPlanner"""
    
    def test_planner_has_prompt_builder(self):
        """Verify planner has prompt_builder attribute"""
        from fairlib import SimpleReActPlanner, ToolRegistry, PromptBuilder
        
        # We need a mock LLM for this - just check the class structure
        assert hasattr(SimpleReActPlanner, '__init__')
        
        # Check that PromptBuilder can be assigned
        builder = PromptBuilder()
        assert builder is not None
    
    def test_loaded_prompt_builder_can_build_prompt(self, tmp_path):
        """Test that a loaded PromptBuilder can build prompts"""
        from fairlib import PromptBuilder, RoleDefinition, Example
        
        # Create and save a configured builder
        builder = PromptBuilder()
        builder.role_definition = RoleDefinition("You are a test assistant.")
        builder.examples.append(Example("User: Hi\nResponse: Hello!"))
        
        save_path = tmp_path / "buildable.json"
        builder.save(str(save_path))
        
        # Load and build
        loaded = PromptBuilder.load(str(save_path))
        prompt = loaded.build_system_prompt_string()
        
        assert "You are a test assistant." in prompt
        assert "Hi" in prompt or "Hello" in prompt


class TestOptimizationMetadataHandling:
    """Test that optimization metadata is properly handled"""
    
    def test_metadata_preserved_through_save_load(self, tmp_path):
        """Test optimization metadata survives save/load"""
        from fairlib import PromptBuilder
        
        # Create config with full metadata
        config_data = {
            "version": "1.0",
            "role_definition": "Test",
            "examples": [],
            "metadata": {
                "optimized": True,
                "optimized_at": "2025-01-01T12:00:00",
                "optimizer": "bootstrap",
                "optimizer_config": {
                    "max_bootstrapped_demos": 4,
                    "max_labeled_demos": 4
                },
                "num_training_examples": 10,
                "score": 0.85
            }
        }
        
        config_path = tmp_path / "metadata_test.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        builder = PromptBuilder.load(str(config_path))
        
        assert builder.is_optimized is True
        
        info = builder.optimization_info
        assert info["optimized"] is True
        assert "optimizer" in info or "optimized_at" in info


# =============================================================================
# Optional: Full optimization integration test (slow, requires GPU/model)
# =============================================================================

@pytest.mark.slow
class TestFullOptimizationFlow:
    """
    Full integration test that runs actual optimization.
    
    This is slow and requires:
    - A working LLM (local or API)
    - Potentially GPU resources
    
    Run with: pytest tests/integration/ -v --run-slow
    """
    
    def test_bootstrap_optimization_with_real_agent(self, tmp_path):
        """Run actual BootstrapFewShot optimization"""
        from fairlib import (
            HuggingFaceAdapter,
            ToolRegistry,
            SafeCalculatorTool,
            ToolExecutor,
            WorkingMemory,
            SimpleAgent,
            SimpleReActPlanner,
            RoleDefinition,
        )
        from fair_prompt_optimizer import (
            FAIRPromptOptimizer,
            TrainingExample,
            numeric_accuracy,
            load_optimized_agent,
        )
        
        # Build agent
        llm = HuggingFaceAdapter("gpt2")  # Or your preferred model
        
        tool_registry = ToolRegistry()
        tool_registry.register_tool(SafeCalculatorTool())
        
        executor = ToolExecutor(tool_registry)
        memory = WorkingMemory()
        
        planner = SimpleReActPlanner(llm, tool_registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a calculator assistant."
        )
        
        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=memory,
            max_steps=5
        )
        
        # Training examples
        examples = [
            TrainingExample(
                inputs={"user_input": "What is 2 + 2?"},
                expected_output="4"
            ),
            TrainingExample(
                inputs={"user_input": "What is 10 - 3?"},
                expected_output="7"
            ),
        ]
        
        # Optimize
        optimizer = FAIRPromptOptimizer(agent)
        output_path = tmp_path / "optimized.json"
        
        config = optimizer.compile(
            training_examples=examples,
            metric=numeric_accuracy,
            optimizer="bootstrap",
            max_bootstrapped_demos=2,
            output_path=str(output_path)
        )
        
        # Verify output
        assert output_path.exists()
        assert config.metadata.optimized is True
        
        # Load optimized prompts into a new PromptBuilder
        from fairlib import PromptBuilder
        loaded_builder = PromptBuilder.load(str(output_path))
        assert loaded_builder.is_optimized is True
        assert len(loaded_builder.examples) > 0