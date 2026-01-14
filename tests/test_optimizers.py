# tests/test_optimizers.py
"""
Unit tests for fair_prompt_optimizer optimizers module.

Tests are organized into:
1. Unit tests (no LLM required) - test structure, config handling, utilities
2. Integration tests (require fairlib) - test with mock agents
3. E2E tests (require LLM) - marked with @pytest.mark.slow

Run with: pytest tests/test_optimizers.py -v
Skip slow tests: pytest tests/test_optimizers.py -v -m "not slow"
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass
import asyncio


# =============================================================================
# Test Utilities
# =============================================================================

class TestRunAsync:
    """Test the run_async utility function."""
    
    def test_run_simple_coroutine(self):
        from fair_prompt_optimizer.optimizers import run_async
        
        async def simple_coro():
            return "hello"
        
        result = run_async(simple_coro())
        assert result == "hello"
    
    def test_run_coroutine_with_await(self):
        from fair_prompt_optimizer.optimizers import run_async
        
        async def coro_with_sleep():
            await asyncio.sleep(0.01)
            return 42
        
        result = run_async(coro_with_sleep())
        assert result == 42
    
    def test_run_coroutine_with_exception(self):
        from fair_prompt_optimizer.optimizers import run_async
        
        async def failing_coro():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            run_async(failing_coro())


class TestClearCudaMemory:
    """Test CUDA memory clearing utility."""
    
    def test_clear_cuda_memory_no_torch(self):
        """Should not raise even if torch is not installed."""
        from fair_prompt_optimizer.optimizers import clear_cuda_memory
        
        # Should not raise
        clear_cuda_memory()
    
    def test_clear_cuda_memory_with_mock_torch(self):
        """Test with mocked torch."""
        from fair_prompt_optimizer.optimizers import clear_cuda_memory
        
        with patch.dict('sys.modules', {'torch': MagicMock()}):
            clear_cuda_memory()  # Should not raise


# =============================================================================
# Test DSPy Module Wrappers
# =============================================================================

class TestSimpleAgentModule:
    """Test SimpleAgentModule DSPy wrapper."""
    
    def test_init(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        mock_agent = Mock()
        module = SimpleAgentModule(mock_agent)
        
        assert module.agent == mock_agent
        assert module.input_field == "user_input"
        assert module.output_field == "response"
    
    def test_init_custom_fields(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        mock_agent = Mock()
        module = SimpleAgentModule(
            mock_agent,
            input_field="query",
            output_field="answer"
        )
        
        assert module.input_field == "query"
        assert module.output_field == "answer"
    
    def test_forward_calls_agent(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.memory = Mock()
        
        # Mock arun to return a result
        async def mock_arun(input_text):
            return f"Response to: {input_text}"
        mock_agent.arun = mock_arun
        
        module = SimpleAgentModule(mock_agent)
        
        # Call forward
        result = module(user_input="Hello")
        
        assert hasattr(result, 'response')
        assert "Hello" in result.response
    
    def test_forward_handles_error(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        
        async def failing_arun(input_text):
            raise RuntimeError("Agent failed")
        mock_agent.arun = failing_arun
        
        module = SimpleAgentModule(mock_agent)
        result = module(user_input="Hello")
        
        assert "Error" in result.response
    
    def test_get_prompt_builder(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        mock_builder = Mock()
        mock_planner = Mock()
        mock_planner.prompt_builder = mock_builder
        mock_agent = Mock()
        mock_agent.planner = mock_planner
        
        module = SimpleAgentModule(mock_agent)
        
        assert module.get_prompt_builder() == mock_builder
    
    def test_get_prompt_builder_none(self):
        from fair_prompt_optimizer.optimizers import SimpleAgentModule
        
        mock_agent = Mock(spec=[])  # No planner attribute
        module = SimpleAgentModule(mock_agent)
        
        assert module.get_prompt_builder() is None


class TestHierarchicalAgentModule:
    """Test HierarchicalAgentModule DSPy wrapper."""
    
    def test_init(self):
        from fair_prompt_optimizer.optimizers import HierarchicalAgentModule
        
        mock_runner = Mock()
        mock_runner.manager = Mock()
        mock_runner.workers = {"Worker1": Mock()}
        
        module = HierarchicalAgentModule(mock_runner)
        
        assert module.runner == mock_runner
    
    def test_get_manager_prompt_builder(self):
        from fair_prompt_optimizer.optimizers import HierarchicalAgentModule
        
        mock_builder = Mock()
        mock_planner = Mock()
        mock_planner.prompt_builder = mock_builder
        
        mock_manager = Mock()
        mock_manager.planner = mock_planner
        
        mock_runner = Mock()
        mock_runner.manager = mock_manager
        mock_runner.workers = {}
        
        module = HierarchicalAgentModule(mock_runner)
        
        assert module.get_manager_prompt_builder() == mock_builder
    
    def test_get_worker_prompt_builders(self):
        from fair_prompt_optimizer.optimizers import HierarchicalAgentModule
        
        mock_builder1 = Mock()
        mock_builder2 = Mock()
        
        mock_worker1 = Mock()
        mock_worker1.planner = Mock()
        mock_worker1.planner.prompt_builder = mock_builder1
        
        mock_worker2 = Mock()
        mock_worker2.planner = Mock()
        mock_worker2.planner.prompt_builder = mock_builder2
        
        mock_runner = Mock()
        mock_runner.manager = Mock()
        mock_runner.workers = {
            "Calculator": mock_worker1,
            "Researcher": mock_worker2,
        }
        
        module = HierarchicalAgentModule(mock_runner)
        builders = module.get_worker_prompt_builders()
        
        assert len(builders) == 2
        assert builders["Calculator"] == mock_builder1
        assert builders["Researcher"] == mock_builder2


class TestSimpleLLMModule:
    """Test SimpleLLMModule DSPy wrapper."""
    
    def test_init(self):
        from fair_prompt_optimizer.optimizers import SimpleLLMModule
        
        mock_llm = Mock()
        module = SimpleLLMModule(mock_llm, "You are helpful.")
        
        assert module.llm == mock_llm
        assert module.system_prompt == "You are helpful."
    
    def test_forward(self):
        from fair_prompt_optimizer.optimizers import SimpleLLMModule
        
        mock_response = Mock()
        mock_response.content = "Hello there!"
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=mock_response)
        
        module = SimpleLLMModule(mock_llm, "You are helpful.")
        result = module(user_input="Hi")
        
        assert result.response == "Hello there!"
        mock_llm.invoke.assert_called_once()


# =============================================================================
# Test AgentOptimizer
# =============================================================================

class TestAgentOptimizer:
    """Test AgentOptimizer class."""
    
    def test_init_creates_config(self):
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import OptimizedConfig
        
        # Mock agent with necessary attributes
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "MockAdapter"
        mock_llm.model_name = "test-model"
        
        mock_builder = Mock()
        mock_builder.role_definition = Mock()
        mock_builder.role_definition.text = "Test role"
        mock_builder.tool_instructions = []
        mock_builder.worker_instructions = []
        mock_builder.format_instructions = []
        mock_builder.examples = []
        
        mock_planner = Mock()
        mock_planner.prompt_builder = mock_builder
        mock_planner.tool_registry = Mock()
        mock_planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_planner.__class__.__name__ = "MockPlanner"
        
        mock_agent = Mock()
        mock_agent.llm = mock_llm
        mock_agent.planner = mock_planner
        mock_agent.__class__.__name__ = "MockAgent"
        mock_agent.max_steps = 10
        mock_agent.stateless = False
        
        # Patch OptimizedConfig.from_agent to return a proper config
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0",
                "type": "agent",
                "prompts": {"role_definition": "Test", "examples": []},
                "model": {},
                "agent": {},
            })
            
            optimizer = AgentOptimizer(mock_agent)
            
            assert optimizer.agent == mock_agent
            assert isinstance(optimizer.config, OptimizedConfig)
            mock_from_agent.assert_called_once_with(mock_agent)
    
    def test_init_with_existing_config(self):
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import OptimizedConfig
        
        mock_agent = Mock()
        existing_config = OptimizedConfig(config={
            "version": "1.0",
            "type": "agent",
            "prompts": {"role_definition": "Existing", "examples": ["ex1"]},
        })
        
        optimizer = AgentOptimizer(mock_agent, config=existing_config)
        
        assert optimizer.config == existing_config
        assert optimizer.config.role_definition == "Existing"
    
    def test_test_method(self):
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import OptimizedConfig
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        
        async def mock_arun(input_text):
            return f"Result: {input_text}"
        mock_agent.arun = mock_arun
        
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0",
                "type": "agent", 
                "prompts": {},
            })
            
            optimizer = AgentOptimizer(mock_agent)
            result = optimizer.test("What is 2+2?")
            
            assert "2+2" in result


class TestSimpleLLMOptimizer:
    """Test SimpleLLMOptimizer class."""
    
    def test_init(self):
        from fair_prompt_optimizer.optimizers import SimpleLLMOptimizer
        
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "MockAdapter"
        mock_llm.model_name = "test-model"
        
        optimizer = SimpleLLMOptimizer(mock_llm, "You are a classifier.")
        
        assert optimizer.llm == mock_llm
        assert optimizer.system_prompt == "You are a classifier."
        assert optimizer.config.type == "simple_llm"
        assert optimizer.config.role_definition == "You are a classifier."
    
    def test_test_method(self):
        from fair_prompt_optimizer.optimizers import SimpleLLMOptimizer
        
        mock_response = Mock()
        mock_response.content = "CATEGORY: positive"
        
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "MockAdapter"
        mock_llm.model_name = "test"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        optimizer = SimpleLLMOptimizer(mock_llm, "Classify sentiment.")
        result = optimizer.test("I love this!")
        
        assert result == "CATEGORY: positive"


# =============================================================================
# Test Optimizer Config Integration
# =============================================================================

class TestOptimizerConfigIntegration:
    """Test that optimizers properly update configs."""
    
    def test_config_examples_accessor(self):
        from fair_prompt_optimizer.config import OptimizedConfig
        
        config = OptimizedConfig(config={
            "prompts": {"examples": ["ex1", "ex2"]}
        })
        
        assert config.examples == ["ex1", "ex2"]
        
        config.examples = ["new1", "new2", "new3"]
        assert len(config.examples) == 3
    
    def test_config_role_definition_accessor(self):
        from fair_prompt_optimizer.config import OptimizedConfig
        
        config = OptimizedConfig(config={
            "prompts": {"role_definition": "Original"}
        })
        
        assert config.role_definition == "Original"
        
        config.role_definition = "Updated"
        assert config.role_definition == "Updated"
    
    def test_provenance_recorded_after_optimization(self):
        """Test that optimization records provenance."""
        from fair_prompt_optimizer.config import OptimizedConfig
        
        config = OptimizedConfig(config={
            "prompts": {"examples": []}
        })
        
        # Simulate what compile() does
        config.optimization.record_run(
            optimizer="bootstrap",
            metric="accuracy",
            num_examples=10,
            examples_before=0,
            examples_after=3,
        )
        
        assert config.optimization.optimized == True
        assert config.optimization.optimizer == "bootstrap"
        assert len(config.optimization.runs) == 1
        assert config.optimization.runs[0].examples_before == 0
        assert config.optimization.runs[0].examples_after == 3


# =============================================================================
# Test Compile Method (with mocked DSPy)
# =============================================================================

class TestCompileWithMockedDSPy:
    """Test compile() method with mocked DSPy optimizers."""
    
    def test_compile_bootstrap_with_manual_success(self):
        """Test compile when manual bootstrapping finds successful demos."""
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        
        async def mock_arun(x):
            return "The answer is 42"
        mock_agent.arun = mock_arun
        
        # Mock the prompt builder
        mock_builder = Mock()
        mock_builder.role_definition = Mock()
        mock_builder.role_definition.text = "Test role"
        mock_builder.tool_instructions = []
        mock_builder.worker_instructions = []
        mock_builder.format_instructions = []
        mock_builder.examples = []
        
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = mock_builder
        
        examples = [
            TrainingExample(inputs={"user_input": "What is 6*7?"}, expected_output="42"),
        ]
        
        # Metric that passes when "42" is in response
        def mock_metric(ex, pred, trace=None):
            return "42" in str(getattr(pred, 'response', ''))
        
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0",
                "type": "agent",
                "prompts": {"role_definition": "Test", "examples": []},
            })
            
            optimizer = AgentOptimizer(mock_agent)
            result = optimizer.compile(
                training_examples=examples,
                metric=mock_metric,
                optimizer="bootstrap",
                max_bootstrapped_demos=3,
            )
            
            # Verify manual bootstrapping captured demos
            assert len(result.examples) > 0
            
            # Verify provenance was recorded
            assert result.optimization.optimized == True
            assert result.optimization.optimizer == "bootstrap"
            assert result.optimization.runs[0].examples_after > 0
    
    def test_compile_bootstrap_falls_back_to_dspy(self):
        """Test that DSPy is called when manual bootstrapping finds nothing."""
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        
        async def mock_arun(x):
            return "wrong answer"
        mock_agent.arun = mock_arun
        
        # Mock the prompt builder
        mock_builder = Mock()
        mock_builder.role_definition = Mock()
        mock_builder.role_definition.text = "Test"
        mock_builder.tool_instructions = []
        mock_builder.worker_instructions = []
        mock_builder.format_instructions = []
        mock_builder.examples = []
        
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = mock_builder
        
        examples = [
            TrainingExample(inputs={"user_input": "2+2"}, expected_output="4"),
        ]
        
        # Metric that always fails (so manual bootstrapping finds nothing)
        def mock_metric(ex, pred, trace=None):
            return False
        
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0",
                "type": "agent",
                "prompts": {"role_definition": "Test", "examples": []},
            })
            
            with patch('dspy.teleprompt.BootstrapFewShot') as MockBootstrap:
                # Setup mock optimizer
                mock_dspy_optimizer = Mock()
                mock_optimized_module = Mock()
                mock_optimized_module.get_prompt_builder = Mock(return_value=mock_builder)
                mock_optimized_module.get_demos = Mock(return_value=[])
                mock_dspy_optimizer.compile = Mock(return_value=mock_optimized_module)
                MockBootstrap.return_value = mock_dspy_optimizer
                
                optimizer = AgentOptimizer(mock_agent)
                result = optimizer.compile(
                    training_examples=examples,
                    metric=mock_metric,
                    optimizer="bootstrap",
                )
                
                # Verify DSPy optimizer was called (fallback)
                MockBootstrap.assert_called_once()
                mock_dspy_optimizer.compile.assert_called_once()
                
                # Verify provenance was recorded
                assert result.optimization.optimized == True
                assert result.optimization.optimizer == "bootstrap"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in optimizers."""
    
    def test_unknown_optimizer_raises(self):
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0", "type": "agent", "prompts": {}
            })
            
            optimizer = AgentOptimizer(mock_agent)
            
            with pytest.raises(ValueError, match="Unknown optimizer"):
                optimizer.compile(
                    training_examples=[TrainingExample(inputs={"user_input": "test"}, expected_output="test")],
                    metric=lambda ex, pred, trace=None: True,  # 3 args!
                    optimizer="invalid_optimizer",
                )
    
    def test_mipro_without_lm_raises(self):
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig
        
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        
        with patch.object(OptimizedConfig, 'from_agent') as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(config={
                "version": "1.0", "type": "agent", "prompts": {}
            })
            
            optimizer = AgentOptimizer(mock_agent)
            
            with pytest.raises(ValueError, match="MIPROv2 requires dspy_lm"):
                optimizer.compile(
                    training_examples=[TrainingExample(inputs={"user_input": "test"}, expected_output="test")],
                    metric=lambda ex, pred, trace=None: True,  # 3 args!
                    optimizer="mipro",
                    dspy_lm=None,
                )


# =============================================================================
# Slow Tests (require actual LLM)
# =============================================================================

@pytest.mark.slow
class TestWithRealLLM:
    """
    Tests that require a real LLM.
    
    Skip with: pytest -m "not slow"
    """
    
    def test_full_optimization_flow(self):
        """Full optimization flow - requires LLM and fairlib."""
        pytest.skip("Requires LLM - run manually")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    mock_llm = Mock()
    mock_llm.__class__.__name__ = "MockAdapter"
    mock_llm.model_name = "test-model"
    
    mock_builder = Mock()
    mock_builder.role_definition = Mock()
    mock_builder.role_definition.text = "Test role"
    mock_builder.tool_instructions = []
    mock_builder.worker_instructions = []
    mock_builder.format_instructions = []
    mock_builder.examples = []
    
    mock_planner = Mock()
    mock_planner.prompt_builder = mock_builder
    
    mock_agent = Mock()
    mock_agent.llm = mock_llm
    mock_agent.planner = mock_planner
    mock_agent.memory = Mock()
    
    async def mock_arun(x):
        return f"Response: {x}"
    mock_agent.arun = mock_arun
    
    return mock_agent


@pytest.fixture
def sample_examples():
    """Create sample training examples."""
    from fair_prompt_optimizer.config import TrainingExample
    
    return [
        TrainingExample(inputs={"user_input": "What is 2+2?"}, expected_output="4"),
        TrainingExample(inputs={"user_input": "What is 3*3?"}, expected_output="9"),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])