# test_fair_agent_module.py
"""
Tests for fair_prompt_optimizer.fair_agent_module module.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import dspy


class TestFAIRAgentModuleInit:
    """Tests for FAIRAgentModule initialization"""
    
    def test_creates_module(self, mock_agent):
        """Test basic module creation"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        module = FAIRAgentModule(mock_agent)
        
        assert module is not None
        assert module.agent is mock_agent
    
    def test_extracts_role_definition(self, mock_agent):
        """Test role definition extraction"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.planner.prompt_builder.role_definition.text = "Test role"
        
        module = FAIRAgentModule(mock_agent)
        
        assert module._initial_role == "Test role"
    
    def test_extracts_model_info(self, mock_agent):
        """Test model info extraction"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.llm.model_name = "test-model-name"
        
        module = FAIRAgentModule(mock_agent)
        
        assert module._model_name == "test-model-name"
    
    def test_extracts_agent_info(self, mock_agent):
        """Test agent info extraction"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.max_steps = 15
        
        module = FAIRAgentModule(mock_agent)
        
        assert module._max_steps == 15
    
    def test_extracts_tools(self, mock_agent):
        """Test tool extraction"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        module = FAIRAgentModule(mock_agent)
        
        # Should extract tool class names
        assert isinstance(module._tools, list)
    
    def test_creates_signature(self, mock_agent):
        """Test DSPy signature creation"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        module = FAIRAgentModule(mock_agent)
        
        assert hasattr(module, 'signature')
        assert issubclass(module.signature, dspy.Signature)
    
    def test_creates_predictor(self, mock_agent):
        """Test DSPy predictor creation"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        module = FAIRAgentModule(mock_agent)
        
        assert hasattr(module, 'predict')


class TestFAIRAgentModuleForward:
    """Tests for FAIRAgentModule.forward method"""
    
    def test_returns_prediction(self, mock_agent):
        """Test that forward returns a DSPy Prediction"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.set_responses(["42"])
        
        module = FAIRAgentModule(mock_agent)
        result = module.forward("What is 6 * 7?")
        
        assert isinstance(result, dspy.Prediction)
        assert result.response == "42"
    
    def test_resets_memory(self, mock_agent):
        """Test that memory is reset before each run"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.memory.set("key", "value")
        
        module = FAIRAgentModule(mock_agent)
        module.forward("test")
        
        # Memory should be cleared
        assert mock_agent.memory.get("key") is None
    
    def test_handles_string_response(self, mock_agent):
        """Test handling of string response"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.set_responses(["Simple string response"])
        
        module = FAIRAgentModule(mock_agent)
        result = module.forward("test")
        
        assert result.response == "Simple string response"
    
    def test_handles_object_response_with_text(self, mock_agent):
        """Test handling of object response with text attribute"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        @dataclass
        class ResponseObject:
            text: str
        
        # Override run to return object
        original_run = mock_agent.run
        mock_agent.run = lambda x: ResponseObject(text="Object response")
        
        module = FAIRAgentModule(mock_agent)
        result = module.forward("test")
        
        assert result.response == "Object response"
        
        # Restore
        mock_agent.run = original_run


class TestFAIRAgentModuleGetConfig:
    """Tests for FAIRAgentModule.get_config method"""
    
    def test_returns_fair_config(self, mock_agent):
        """Test that get_config returns FAIRConfig"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        from fair_prompt_optimizer.translator import FAIRConfig
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert isinstance(config, FAIRConfig)
    
    def test_includes_model_config(self, mock_agent):
        """Test that config includes model info"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.llm.model_name = "config-test-model"
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert config.model.model_name == "config-test-model"
    
    def test_includes_agent_config(self, mock_agent):
        """Test that config includes agent info"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.max_steps = 20
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert config.agent.max_steps == 20
    
    def test_includes_role_definition(self, mock_agent):
        """Test that config includes role definition"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.planner.prompt_builder.role_definition.text = "Config role"
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert config.role_definition == "Config role"


class TestFAIRPromptOptimizerInit:
    """Tests for FAIRPromptOptimizer initialization"""
    
    def test_creates_optimizer(self, mock_agent):
        """Test basic optimizer creation"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        assert optimizer is not None
        assert optimizer.agent is mock_agent
    
    def test_module_not_created_until_compile(self, mock_agent):
        """Test that module is created during compile, not init"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        assert optimizer._module is None


class TestFAIRPromptOptimizerCompile:
    """Tests for FAIRPromptOptimizer.compile method"""
    
    def test_bootstrap_compilation(
        self, mock_agent, sample_training_examples
    ):
        """Test compilation with BootstrapFewShot"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        # Set up agent to return correct answers
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            # Mock the optimizer
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = MagicMock(
                role_definition="Optimized",
                examples=[],
                model=MagicMock(model_name="test", adapter="Test", adapter_kwargs={}),
                agent=MagicMock(agent_type="Test", planner_type="Test", max_steps=10, tools=[]),
                metadata=None
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap",
                max_bootstrapped_demos=2
            )
            
            assert config is not None
            mock_bootstrap.assert_called_once()
    
    def test_mipro_requires_lm(self, mock_agent, sample_training_examples):
        """Test that MIPROv2 requires a DSPy LM"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with pytest.raises(ValueError, match="MIPROv2 requires a DSPy LM"):
            optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="mipro",
                dspy_lm=None  # No LM provided
            )
    
    def test_unknown_optimizer_raises(
        self, mock_agent, sample_training_examples
    ):
        """Test that unknown optimizer raises error"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with pytest.raises(ValueError, match="Unknown optimizer"):
            optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="unknown_optimizer"
            )
    
    def test_saves_config_to_path(
        self, mock_agent, sample_training_examples, tmp_path
    ):
        """Test that config is saved when output_path provided"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig, ModelConfig, AgentConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        output_path = tmp_path / "optimized.json"
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            # Return a proper FAIRConfig object, not a MagicMock
            mock_compiled.get_config.return_value = FAIRConfig(
                role_definition="Optimized",
                examples=[],
                model=ModelConfig(model_name="test", adapter="TestAdapter"),
                agent=AgentConfig(agent_type="SimpleAgent", planner_type="SimpleReActPlanner", max_steps=10, tools=[])
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap",
                output_path=str(output_path)
            )
            
            assert output_path.exists()
    
    def test_adds_optimization_metadata(
        self, mock_agent, sample_training_examples
    ):
        """Test that optimization metadata is added"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig(
                role_definition="Optimized"
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            assert config.metadata is not None
            assert config.metadata.optimized is True
            assert config.metadata.optimizer == "bootstrap"
            assert config.metadata.num_training_examples == 3
    
    def test_converts_training_examples_to_dspy(
        self, mock_agent, sample_training_examples
    ):
        """Test that TrainingExamples are converted to DSPy format"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            # Check that compile was called with DSPy examples
            call_args = mock_bootstrap.return_value.compile.call_args
            trainset = call_args.kwargs.get('trainset') or call_args[1].get('trainset')
            
            assert len(trainset) == 3
            assert hasattr(trainset[0], 'user_input')


class TestFAIRPromptOptimizerMIPRO:
    """Tests for MIPROv2 optimization"""
    
    def test_mipro_with_lm(self, mock_agent, sample_training_examples):
        """Test MIPROv2 compilation with DSPy LM"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        mock_lm = MagicMock()
        
        with patch('dspy.teleprompt.MIPROv2') as mock_mipro:
            with patch('dspy.configure') as mock_configure:
                mock_compiled = MagicMock()
                mock_compiled.get_config.return_value = FAIRConfig()
                mock_mipro.return_value.compile.return_value = mock_compiled
                
                config = optimizer.compile(
                    training_examples=sample_training_examples,
                    metric=numeric_accuracy,
                    optimizer="mipro",
                    mipro_auto="light",
                    dspy_lm=mock_lm
                )
                
                # Verify DSPy was configured with the LM
                mock_configure.assert_called_once_with(lm=mock_lm)
                mock_mipro.assert_called_once()
    
    def test_mipro_modes(self, mock_agent, sample_training_examples):
        """Test different MIPROv2 modes"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"] * 3)  # Multiple runs
        mock_lm = MagicMock()
        
        for mode in ["light", "medium", "heavy"]:
            optimizer = FAIRPromptOptimizer(mock_agent)
            
            with patch('dspy.teleprompt.MIPROv2') as mock_mipro:
                with patch('dspy.configure'):
                    mock_compiled = MagicMock()
                    mock_compiled.get_config.return_value = FAIRConfig()
                    mock_mipro.return_value.compile.return_value = mock_compiled
                    
                    config = optimizer.compile(
                        training_examples=sample_training_examples,
                        metric=numeric_accuracy,
                        optimizer="mipro",
                        mipro_auto=mode,
                        dspy_lm=mock_lm
                    )
                    
                    # Verify mode was passed
                    mock_mipro.assert_called_with(
                        metric=numeric_accuracy,
                        auto=mode
                    )


class TestFAIRPromptOptimizerIntegration:
    """Integration tests for FAIRPromptOptimizer"""
    
    def test_full_optimization_flow(
        self, mock_agent, sample_training_examples, tmp_path
    ):
        """Test complete optimization flow"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import load_fair_config, FAIRConfig
        
        # Set up mock agent responses
        mock_agent.set_responses(["42", "25", "56"])
        mock_agent.planner.prompt_builder.role_definition.text = "Math assistant"
        mock_agent.llm.model_name = "test-model"
        mock_agent.max_steps = 10
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        output_path = tmp_path / "optimized.json"
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig(
                role_definition="Math assistant",
                examples=["Demo 1"]
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap",
                max_bootstrapped_demos=4,
                output_path=str(output_path)
            )
            
            # Verify config file was created
            assert output_path.exists()
            
            # Verify config can be loaded
            loaded = load_fair_config(str(output_path))
            assert loaded.metadata.optimized is True
            assert loaded.metadata.optimizer == "bootstrap"


class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_empty_training_examples(self, mock_agent):
        """Test handling of empty training examples"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=[],
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            assert config.metadata.num_training_examples == 0
    
    def test_agent_without_role_definition(self, mock_llm, mock_tool_registry, mock_memory):
        """Test handling of agent without role definition"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        from tests.conftest import MockPlanner, MockToolExecutor, MockAgent
        
        # Create planner without role definition
        planner = MockPlanner(mock_llm, mock_tool_registry)
        delattr(planner.prompt_builder, 'role_definition')
        
        executor = MockToolExecutor(mock_tool_registry)
        agent = MockAgent(mock_llm, planner, executor, mock_memory)
        
        # Should use default role
        module = FAIRAgentModule(agent)
        assert module._initial_role == "Complete the given task."
    
    def test_custom_metric_function(
        self, mock_agent, sample_training_examples
    ):
        """Test using a custom metric function"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        # Custom metric
        def custom_metric(example, prediction, trace=None):
            return "4" in prediction.response
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=custom_metric,
                optimizer="bootstrap"
            )
            
            # Verify custom metric was passed
            call_kwargs = mock_bootstrap.call_args.kwargs
            assert call_kwargs['metric'] is custom_metric