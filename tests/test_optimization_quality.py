# test_optimization_quality.py
"""
Tests for optimization quality and effectiveness.
"""

from unittest.mock import MagicMock, patch
from datetime import datetime

from fair_prompt_optimizer.translator import (
    FAIRConfig,
    ModelConfig,
    AgentConfig
)


class TestConfigCompleteness:
    """Tests that optimized configs are complete and valid"""
    
    def test_optimized_config_has_all_required_fields(self, mock_agent, sample_training_examples):
        """Test that optimized config includes all required fields"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        mock_agent.set_responses(["42", "25", "56"])
        mock_agent.planner.prompt_builder.role_definition.text = "Test role"
        mock_agent.llm.model_name = "test-model"
        mock_agent.max_steps = 10
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig(
                role_definition="Test role",
                model=ModelConfig(model_name="test-model", adapter="TestAdapter"),
                agent=AgentConfig(agent_type="SimpleAgent", max_steps=10, tools=[])
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            # Verify all required fields are present
            assert config.version is not None
            assert config.model is not None
            assert config.model.model_name is not None
            assert config.model.adapter is not None
            assert config.agent is not None
            assert config.agent.agent_type is not None
            assert config.agent.planner_type is not None
            assert config.agent.max_steps is not None
    
    def test_metadata_is_populated(self, mock_agent, sample_training_examples):
        """Test that optimization metadata is populated"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            assert config.metadata is not None
            assert config.metadata.optimized is True
            assert config.metadata.optimizer == "bootstrap"
            assert config.metadata.optimized_at is not None
            assert config.metadata.num_training_examples == 3
    
    def test_optimizer_config_is_recorded(self, mock_agent, sample_training_examples):
        """Test that optimizer configuration is recorded in metadata"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap",
                max_bootstrapped_demos=5,
                max_labeled_demos=3
            )
            
            assert config.metadata.optimizer_config is not None
            assert config.metadata.optimizer_config["max_bootstrapped_demos"] == 5
            assert config.metadata.optimizer_config["max_labeled_demos"] == 3


class TestDemoQuality:
    """Tests for quality of generated demos"""
    
    def test_demos_are_strings(self, mock_agent, sample_training_examples):
        """Test that generated demos are strings"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        from fair_prompt_optimizer.translator import FAIRConfig
        import dspy
        
        mock_agent.set_responses(["42"])
        mock_agent.planner.prompt_builder.role_definition.text = "Test"
        
        module = FAIRAgentModule(mock_agent)
        
        # Simulate having demos
        mock_demo = dspy.Example(
            user_input="What is 2+2?",
            response="4"
        )
        module.predict.demos = [mock_demo]
        
        config = module.get_config()
        
        for example in config.examples:
            assert isinstance(example, str)
    
    def test_demo_format_includes_user_and_response(self, mock_agent):
        """Test that demos include User and Response sections"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        import dspy
        
        mock_agent.planner.prompt_builder.role_definition.text = "Test"
        
        module = FAIRAgentModule(mock_agent)
        
        # Simulate having demos
        mock_demo = dspy.Example(
            user_input="Test question",
            response="Test answer"
        )
        module.predict.demos = [mock_demo]
        
        config = module.get_config()
        
        if config.examples:
            demo = config.examples[0]
            assert "User:" in demo or "user_input" in demo.lower()
            assert "Response:" in demo or "response" in demo.lower()


class TestConfigRoundtrip:
    """Tests for config save/load roundtrip after optimization"""
    
    def test_optimized_config_can_be_saved_and_loaded(
        self, mock_agent, sample_training_examples, tmp_path
    ):
        """Test that optimized config survives save/load cycle"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import load_fair_config
        
        mock_agent.set_responses(["42", "25", "56"])
        mock_agent.planner.prompt_builder.role_definition.text = "Math helper"
        mock_agent.llm.model_name = "test-model"
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        output_path = tmp_path / "optimized.json"
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig(
                role_definition="Math helper",
                examples=["User: Test\nResponse: Answer"],
                model=ModelConfig(model_name="test-model", adapter="TestAdapter"),
                agent=AgentConfig(max_steps=10, tools=["SafeCalculatorTool"])
            )
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            original_config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap",
                output_path=str(output_path)
            )
            
            # Load the saved config
            loaded_config = load_fair_config(str(output_path))
            
            # Verify key fields match
            assert loaded_config.role_definition == original_config.role_definition
            assert len(loaded_config.examples) == len(original_config.examples)
            assert loaded_config.model.model_name == original_config.model.model_name
            assert loaded_config.metadata.optimized == original_config.metadata.optimized


class TestMetricSelection:
    """Tests for metric selection and application"""
    
    def test_correct_metric_passed_to_optimizer(
        self, mock_agent, sample_training_examples
    ):
        """Test that the selected metric is passed to DSPy optimizer"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import exact_match, contains_answer, numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        for metric in [exact_match, contains_answer, numeric_accuracy]:
            optimizer = FAIRPromptOptimizer(mock_agent)
            
            with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
                mock_compiled = MagicMock()
                mock_compiled.get_config.return_value = FAIRConfig()
                mock_bootstrap.return_value.compile.return_value = mock_compiled
                
                optimizer.compile(
                    training_examples=sample_training_examples,
                    metric=metric,
                    optimizer="bootstrap"
                )
                
                # Verify the metric was passed
                call_kwargs = mock_bootstrap.call_args.kwargs
                assert call_kwargs['metric'] is metric


class TestTrainingExampleConversion:
    """Tests for training example conversion quality"""
    
    def test_all_examples_converted(self, mock_agent, sample_training_examples):
        """Test that all training examples are converted"""
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
            
            # Get the trainset that was passed
            call_args = mock_bootstrap.return_value.compile.call_args
            trainset = call_args.kwargs.get('trainset') or call_args[1].get('trainset')
            
            assert len(trainset) == len(sample_training_examples)
    
    def test_example_fields_preserved(self):
        """Test that example fields are preserved during conversion"""
        from fair_prompt_optimizer.translator import DSPyTranslator, TrainingExample
        
        translator = DSPyTranslator()
        
        examples = [
            TrainingExample(
                inputs={"user_input": "Test input"},
                expected_output="Test output"
            )
        ]
        
        dspy_examples = translator.training_examples_to_dspy(examples)
        
        assert dspy_examples[0].user_input == "Test input"
        assert dspy_examples[0].response == "Test output"


class TestAgentPreservation:
    """Tests that agent configuration is preserved through optimization"""
    
    def test_model_name_preserved(self, mock_agent, sample_training_examples):
        """Test that model name is preserved"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.llm.model_name = "specific-model-v1"
        mock_agent.planner.prompt_builder.role_definition.text = "Test"
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert config.model.model_name == "specific-model-v1"
    
    def test_max_steps_preserved(self, mock_agent, sample_training_examples):
        """Test that max_steps is preserved"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        
        mock_agent.max_steps = 25
        mock_agent.planner.prompt_builder.role_definition.text = "Test"
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert config.agent.max_steps == 25
    
    def test_tools_preserved(self, mock_agent, sample_training_examples):
        """Test that tools are preserved"""
        from fair_prompt_optimizer.fair_agent_module import FAIRAgentModule
        from tests.conftest import MockSafeCalculatorTool
        
        # Add a tool to the registry
        mock_agent.planner.tool_registry.register_tool(MockSafeCalculatorTool())
        mock_agent.planner.prompt_builder.role_definition.text = "Test"
        
        module = FAIRAgentModule(mock_agent)
        config = module.get_config()
        
        assert len(config.agent.tools) > 0
        assert "MockSafeCalculatorTool" in config.agent.tools


class TestOptimizationTimestamp:
    """Tests for optimization timestamp handling"""
    
    def test_timestamp_is_iso_format(self, mock_agent, sample_training_examples):
        """Test that optimization timestamp is in ISO format"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            # Should be parseable as ISO format
            timestamp = config.metadata.optimized_at
            parsed = datetime.fromisoformat(timestamp)
            assert parsed is not None
    
    def test_timestamp_is_recent(self, mock_agent, sample_training_examples):
        """Test that optimization timestamp is recent"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        before = datetime.now()
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig()
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            after = datetime.now()
            timestamp = datetime.fromisoformat(config.metadata.optimized_at)
            
            assert before <= timestamp <= after


class TestVersioning:
    """Tests for config versioning"""
    
    def test_config_has_version(self, mock_agent, sample_training_examples):
        """Test that config includes version"""
        from fair_prompt_optimizer.fair_agent_module import FAIRPromptOptimizer
        from fair_prompt_optimizer.metrics import numeric_accuracy
        from fair_prompt_optimizer.translator import FAIRConfig
        
        mock_agent.set_responses(["42", "25", "56"])
        
        optimizer = FAIRPromptOptimizer(mock_agent)
        
        with patch('dspy.teleprompt.BootstrapFewShot') as mock_bootstrap:
            mock_compiled = MagicMock()
            mock_compiled.get_config.return_value = FAIRConfig(version="1.0")
            mock_bootstrap.return_value.compile.return_value = mock_compiled
            
            config = optimizer.compile(
                training_examples=sample_training_examples,
                metric=numeric_accuracy,
                optimizer="bootstrap"
            )
            
            assert config.version is not None
            assert config.version == "1.0"