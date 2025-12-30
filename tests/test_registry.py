# test_registry.py
"""
Tests for fair_prompt_optimizer.registry module.
"""

import pytest

from fair_prompt_optimizer.translator import (
    FAIRConfig,
    ModelConfig,
    AgentConfig,
    save_fair_config,
)


class TestRegistryPopulation:
    """Tests for registry population"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_register_adapter(self):
        """Test registering a custom adapter"""
        from fair_prompt_optimizer.registry import (
            register_adapter, ADAPTER_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        class CustomAdapter:
            pass
        
        register_adapter("CustomAdapter", CustomAdapter)
        assert "CustomAdapter" in ADAPTER_REGISTRY
        assert ADAPTER_REGISTRY["CustomAdapter"] is CustomAdapter
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_register_tool(self):
        """Test registering a custom tool"""
        from fair_prompt_optimizer.registry import (
            register_tool, TOOL_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        class CustomTool:
            pass
        
        register_tool("CustomTool", CustomTool)
        assert "CustomTool" in TOOL_REGISTRY
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_register_planner(self):
        """Test registering a custom planner"""
        from fair_prompt_optimizer.registry import (
            register_planner, PLANNER_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        class CustomPlanner:
            pass
        
        register_planner("CustomPlanner", CustomPlanner)
        assert "CustomPlanner" in PLANNER_REGISTRY
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_register_agent(self):
        """Test registering a custom agent"""
        from fair_prompt_optimizer.registry import (
            register_agent, AGENT_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        class CustomAgent:
            pass
        
        register_agent("CustomAgent", CustomAgent)
        assert "CustomAgent" in AGENT_REGISTRY


class TestPopulateRegistries:
    """Tests for _populate_registries function"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_populates_adapters(self):
        """Test that adapters are populated"""
        from fair_prompt_optimizer.registry import (
            ADAPTER_REGISTRY, _populate_registries
        )
        
        # Clear and repopulate
        ADAPTER_REGISTRY.clear()
        _populate_registries()
        
        assert "HuggingFaceAdapter" in ADAPTER_REGISTRY
        assert "OpenAIAdapter" in ADAPTER_REGISTRY
        assert "AnthropicAdapter" in ADAPTER_REGISTRY

    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_populates_tools(self):
        """Test that tools are populated"""
        from fair_prompt_optimizer.registry import (
            TOOL_REGISTRY, _populate_registries
        )
        
        # Tools may already be populated, just verify it contains expected
        _populate_registries()
        
        # The mock includes SafeCalculatorTool
        assert "SafeCalculatorTool" in TOOL_REGISTRY
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_populates_planners(self):
        """Test that planners are populated"""
        from fair_prompt_optimizer.registry import (
            PLANNER_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        assert "SimpleReActPlanner" in PLANNER_REGISTRY
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_populates_agents(self):
        """Test that agents are populated"""
        from fair_prompt_optimizer.registry import (
            AGENT_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        
        assert "SimpleAgent" in AGENT_REGISTRY
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_idempotent(self):
        """Test that multiple calls don't cause issues"""
        from fair_prompt_optimizer.registry import (
            ADAPTER_REGISTRY, _populate_registries
        )
        
        _populate_registries()
        count1 = len(ADAPTER_REGISTRY)
        
        _populate_registries()
        count2 = len(ADAPTER_REGISTRY)
        
        assert count1 == count2


class TestCreateAgentFromConfig:
    """Tests for create_agent_from_config function"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_creates_agent_with_defaults(self):
        """Test creating agent with default config"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                max_steps=10,
                tools=["SafeCalculatorTool"]
            )
        )
        
        agent = create_agent_from_config(config)
        
        assert agent is not None
        assert agent.max_steps == 10
        assert agent.llm.model_name == "test-model"
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_applies_role_definition(self):
        """Test that role definition is applied"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            role_definition="You are a test assistant.",
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=[]
            )
        )
        
        agent = create_agent_from_config(config)
        
        assert agent.planner.prompt_builder.role_definition.text == "You are a test assistant."
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_registers_tools(self):
        """Test that tools are registered"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=["SafeCalculatorTool"]
            )
        )
        
        agent = create_agent_from_config(config)
        
        # Check that tool was registered
        tool_registry = agent.planner.tool_registry
        assert tool_registry.get_tool("calculator") is not None
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_applies_examples(self):
        """Test that examples are applied to prompt builder"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            examples=["User: Test\nResponse: Test response"],
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=[]
            )
        )
        
        agent = create_agent_from_config(config)
        
        assert len(agent.planner.prompt_builder.examples) == 1
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_adapter_kwargs_passed(self):
        """Test that adapter kwargs are passed through"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter",
                adapter_kwargs={"temperature": 0.7}
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=[]
            )
        )
        
        agent = create_agent_from_config(config)
        
        # Verify adapter was created (kwargs stored internally)
        assert agent.llm is not None


class TestLoadOptimizedAgent:
    """Tests for load_optimized_agent function"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_loads_from_file(self, tmp_path):
        """Test loading agent from config file"""
        from fair_prompt_optimizer.registry import load_optimized_agent
        
        config = FAIRConfig(
            role_definition="Test role",
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=["SafeCalculatorTool"]
            )
        )
        
        config_path = tmp_path / "config.json"
        save_fair_config(config, str(config_path))
        
        agent = load_optimized_agent(str(config_path))
        
        assert agent is not None
        assert agent.planner.prompt_builder.role_definition.text == "Test role"
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_loads_optimized_config(self, tmp_path):
        """Test loading config with optimization metadata"""
        from fair_prompt_optimizer.registry import load_optimized_agent
        from fair_prompt_optimizer.translator import OptimizationMetadata
        
        config = FAIRConfig(
            role_definition="Optimized role",
            examples=["Demo 1", "Demo 2"],
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=[]
            ),
            metadata=OptimizationMetadata(
                optimized=True,
                optimizer="bootstrap"
            )
        )
        
        config_path = tmp_path / "optimized.json"
        save_fair_config(config, str(config_path))
        
        agent = load_optimized_agent(str(config_path))
        
        assert len(agent.planner.prompt_builder.examples) == 2


class TestRegistryErrorHandling:
    """Tests for error handling in registry"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_unknown_adapter_raises(self):
        """Test that unknown adapter raises KeyError"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="NonExistentAdapter"  # Unknown adapter
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=[]
            )
        )
        
        with pytest.raises(KeyError):
            create_agent_from_config(config)
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_unknown_tool_raises(self):
        """Test that unknown tool raises KeyError"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                tools=["NonExistentTool"]  # Unknown tool
            )
        )
        
        with pytest.raises(KeyError):
            create_agent_from_config(config)
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_unknown_planner_raises(self):
        """Test that unknown planner raises KeyError"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="NonExistentPlanner",  # Unknown planner
                tools=[]
            )
        )
        
        with pytest.raises(KeyError):
            create_agent_from_config(config)
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_unknown_agent_type_raises(self):
        """Test that unknown agent type raises KeyError"""
        from fair_prompt_optimizer.registry import create_agent_from_config
        
        config = FAIRConfig(
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter"
            ),
            agent=AgentConfig(
                agent_type="NonExistentAgent",  # Unknown agent
                planner_type="SimpleReActPlanner",
                tools=[]
            )
        )
        
        with pytest.raises(KeyError):
            create_agent_from_config(config)
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_missing_file_raises(self, tmp_path):
        """Test that missing config file raises error"""
        from fair_prompt_optimizer.registry import load_optimized_agent
        
        with pytest.raises(FileNotFoundError):
            load_optimized_agent(str(tmp_path / "nonexistent.json"))


class TestRegistryIntegration:
    """Integration tests for registry functionality"""
    
    @pytest.mark.usefixtures("mock_fairlib_module")
    def test_full_roundtrip(self, tmp_path):
        """Test full config save -> load -> agent creation roundtrip"""
        from fair_prompt_optimizer.registry import load_optimized_agent
        from fair_prompt_optimizer.translator import OptimizationMetadata
        
        # Create a complete config
        config = FAIRConfig(
            version="1.0",
            role_definition="You are a helpful math assistant.",
            examples=[
                "User: What is 2+2?\nResponse: 4",
                "User: What is 3*3?\nResponse: 9"
            ],
            model=ModelConfig(
                model_name="test-model",
                adapter="HuggingFaceAdapter",
                adapter_kwargs={}
            ),
            agent=AgentConfig(
                agent_type="SimpleAgent",
                planner_type="SimpleReActPlanner",
                max_steps=15,
                tools=["SafeCalculatorTool"]
            ),
            metadata=OptimizationMetadata(
                optimized=True,
                optimizer="bootstrap",
                num_training_examples=10
            )
        )
        
        # Save config
        config_path = tmp_path / "full_config.json"
        save_fair_config(config, str(config_path))
        
        # Load agent from config
        agent = load_optimized_agent(str(config_path))
        
        # Verify all components
        assert agent.max_steps == 15
        assert agent.llm.model_name == "test-model"
        assert len(agent.planner.prompt_builder.examples) == 2
        assert agent.planner.prompt_builder.role_definition.text == "You are a helpful math assistant."