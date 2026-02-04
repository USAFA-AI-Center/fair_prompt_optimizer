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

class TestAgentModule:
    """Test AgentModule DSPy wrapper."""

    def test_init(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        mock_agent = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        module = AgentModule(mock_agent)

        assert module.agent == mock_agent
        assert module.input_field == "user_input"
        assert module.output_field == "response"

    def test_init_custom_fields(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        mock_agent = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        module = AgentModule(
            mock_agent,
            input_field="query",
            output_field="answer"
        )

        assert module.input_field == "query"
        assert module.output_field == "answer"

    def test_forward_calls_agent(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        # Create mock agent
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        # Mock arun to return a result
        async def mock_arun(input_text):
            return f"Response to: {input_text}"
        mock_agent.arun = mock_arun

        module = AgentModule(mock_agent)

        # Call forward
        result = module(user_input="Hello")

        assert hasattr(result, 'response')
        assert "Hello" in result.response

    def test_forward_handles_error(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        async def failing_arun(input_text):
            raise RuntimeError("Agent failed")
        mock_agent.arun = failing_arun

        module = AgentModule(mock_agent)
        result = module(user_input="Hello")

        assert "Error" in result.response

    def test_get_prompt_builder(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        mock_builder = Mock()
        mock_planner = Mock()
        mock_planner.prompt_builder = mock_builder
        mock_planner.prompt_builder.role_definition = None
        mock_planner.prompt_builder.format_instructions = []
        mock_planner.tool_registry = Mock()
        mock_planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent = Mock()
        mock_agent.planner = mock_planner
        mock_agent.llm = Mock()

        module = AgentModule(mock_agent)

        assert module.get_prompt_builder() == mock_builder

    def test_get_prompt_builder_none(self):
        from fair_prompt_optimizer.optimizers import AgentModule

        mock_agent = Mock(spec=[])  # No planner attribute
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        module = AgentModule(mock_agent)
        # Now remove the planner for the actual test
        del mock_agent.planner

        assert module.get_prompt_builder() is None


class TestMultiAgentModule:
    """Test MultiAgentModule DSPy wrapper."""

    def test_init(self):
        from fair_prompt_optimizer.optimizers import MultiAgentModule

        mock_runner = Mock()
        mock_runner.manager = Mock()
        mock_runner.manager.planner = Mock()
        mock_runner.manager.planner.prompt_builder = Mock()
        mock_runner.manager.planner.prompt_builder.role_definition = None
        mock_runner.workers = {"Worker1": Mock()}

        module = MultiAgentModule(mock_runner)

        assert module.runner == mock_runner

    def test_get_manager_builder(self):
        from fair_prompt_optimizer.optimizers import MultiAgentModule

        mock_builder = Mock()
        mock_planner = Mock()
        mock_planner.prompt_builder = mock_builder

        mock_manager = Mock()
        mock_manager.planner = mock_planner
        mock_manager.planner.prompt_builder.role_definition = None

        mock_runner = Mock()
        mock_runner.manager = mock_manager
        mock_runner.workers = {}

        module = MultiAgentModule(mock_runner)

        assert module.get_manager_builder() == mock_builder

    def test_get_worker_builders(self):
        from fair_prompt_optimizer.optimizers import MultiAgentModule

        mock_builder1 = Mock()
        mock_builder2 = Mock()

        mock_worker1 = Mock()
        mock_worker1.planner = Mock()
        mock_worker1.planner.prompt_builder = mock_builder1

        mock_worker2 = Mock()
        mock_worker2.planner = Mock()
        mock_worker2.planner.prompt_builder = mock_builder2

        mock_manager = Mock()
        mock_manager.planner = Mock()
        mock_manager.planner.prompt_builder = Mock()
        mock_manager.planner.prompt_builder.role_definition = None

        mock_runner = Mock()
        mock_runner.manager = mock_manager
        mock_runner.workers = {
            "Calculator": mock_worker1,
            "Researcher": mock_worker2,
        }

        module = MultiAgentModule(mock_runner)
        builders = module.get_worker_builders()

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
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

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


class TestMultiAgentOptimizer:
    """Test MultiAgentOptimizer class with worker optimization."""

    def _create_mock_runner(self):
        """Create a mock HierarchicalAgentRunner for testing."""
        # Create mock workers
        mock_worker1 = Mock()
        mock_worker1.memory = Mock()
        mock_worker1.planner = Mock()
        mock_worker1.planner.prompt_builder = Mock()
        mock_worker1.planner.prompt_builder.role_definition = Mock()
        mock_worker1.planner.prompt_builder.role_definition.text = "Worker 1 role"
        mock_worker1.planner.prompt_builder.format_instructions = []
        mock_worker1.planner.prompt_builder.examples = []
        mock_worker1.planner.tool_registry = Mock()
        mock_worker1.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_worker1.llm = Mock()
        mock_worker1.llm.__class__.__name__ = "MockAdapter"
        mock_worker1.llm.model_name = "test-model"
        mock_worker1.__class__.__name__ = "SimpleAgent"
        mock_worker1.max_steps = 5
        mock_worker1.stateless = False

        async def mock_arun1(x):
            return f"Worker1 response: {x}"
        mock_worker1.arun = mock_arun1

        mock_worker2 = Mock()
        mock_worker2.memory = Mock()
        mock_worker2.planner = Mock()
        mock_worker2.planner.prompt_builder = Mock()
        mock_worker2.planner.prompt_builder.role_definition = Mock()
        mock_worker2.planner.prompt_builder.role_definition.text = "Worker 2 role"
        mock_worker2.planner.prompt_builder.format_instructions = []
        mock_worker2.planner.prompt_builder.examples = []
        mock_worker2.planner.tool_registry = Mock()
        mock_worker2.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_worker2.llm = Mock()
        mock_worker2.llm.__class__.__name__ = "MockAdapter"
        mock_worker2.llm.model_name = "test-model"
        mock_worker2.__class__.__name__ = "SimpleAgent"
        mock_worker2.max_steps = 5
        mock_worker2.stateless = False

        async def mock_arun2(x):
            return f"Worker2 response: {x}"
        mock_worker2.arun = mock_arun2

        # Create mock manager
        mock_manager = Mock()
        mock_manager.memory = Mock()
        mock_manager.planner = Mock()
        mock_manager.planner.prompt_builder = Mock()
        mock_manager.planner.prompt_builder.role_definition = Mock()
        mock_manager.planner.prompt_builder.role_definition.text = "Manager role"
        mock_manager.planner.prompt_builder.format_instructions = []
        mock_manager.planner.prompt_builder.examples = []
        mock_manager.planner.tool_registry = Mock()
        mock_manager.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_manager.llm = Mock()

        async def mock_manager_run(x):
            return f"Manager response: {x}"
        mock_manager.arun = mock_manager_run

        # Create mock runner
        mock_runner = Mock()
        mock_runner.manager = mock_manager
        mock_runner.workers = {
            "DataGatherer": mock_worker1,
            "Summarizer": mock_worker2,
        }

        async def mock_run(x):
            return f"Runner response: {x}"
        mock_runner.run = mock_run

        return mock_runner

    def test_init_with_optimize_workers(self):
        """Test initialization with optimize_workers flag."""
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from fair_prompt_optimizer.config import OptimizedConfig

        mock_runner = self._create_mock_runner()

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(
                mock_runner,
                optimize_manager=True,
                optimize_workers=True,
            )

            assert optimizer.optimize_manager == True
            assert optimizer.optimize_workers == True
            assert optimizer.runner == mock_runner

    def test_compile_with_worker_training_examples(self):
        """Test compile() with worker training data."""
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig

        mock_runner = self._create_mock_runner()

        # Training examples for manager
        manager_examples = [
            TrainingExample(
                inputs={"user_input": "Research quantum computing"},
                expected_output="Quantum computing summary",
                full_trace="Manager trace..."
            ),
        ]

        # Training examples for workers
        worker_training_examples = {
            "DataGatherer": [
                TrainingExample(
                    inputs={"user_input": "search quantum computing"},
                    expected_output="Search results",
                    full_trace="DataGatherer trace..."
                ),
            ],
            "Summarizer": [
                TrainingExample(
                    inputs={"user_input": "summarize data"},
                    expected_output="Summary",
                    full_trace="Summarizer trace..."
                ),
            ],
        }

        def mock_metric(ex, pred, trace=None):
            return True

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(
                mock_runner,
                optimize_manager=True,
                optimize_workers=True,
            )

            # Mock the _optimize_workers method to return expected worker configs
            mock_worker_configs = {
                "DataGatherer": {"prompts": {"role_definition": "Optimized DG", "examples": ["ex1"]}},
                "Summarizer": {"prompts": {"role_definition": "Optimized S", "examples": ["ex2"]}},
            }
            with patch.object(optimizer, '_optimize_workers', return_value=mock_worker_configs) as mock_opt_workers:
                result = optimizer.compile(
                    training_examples=manager_examples,
                    metric=mock_metric,
                    worker_training_examples=worker_training_examples,
                    optimizer="bootstrap",
                    max_bootstrapped_demos=2,
                )

                # Verify _optimize_workers was called with correct args
                mock_opt_workers.assert_called_once()
                call_kwargs = mock_opt_workers.call_args.kwargs
                assert call_kwargs['worker_training_examples'] == worker_training_examples

                # Verify workers section exists in config
                assert "workers" in result.config
                assert "DataGatherer" in result.config["workers"]
                assert "Summarizer" in result.config["workers"]

    def test_compile_skips_workers_without_training_data(self):
        """Test that workers without training data are skipped."""
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig

        mock_runner = self._create_mock_runner()

        manager_examples = [
            TrainingExample(
                inputs={"user_input": "Test"},
                expected_output="Result",
                full_trace="Trace..."
            ),
        ]

        # Only provide training data for one worker
        worker_training_examples = {
            "DataGatherer": [
                TrainingExample(
                    inputs={"user_input": "search"},
                    expected_output="results",
                    full_trace="trace..."
                ),
            ],
            # Summarizer has no training data
        }

        def mock_metric(ex, pred, trace=None):
            return True

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(
                mock_runner,
                optimize_manager=True,
                optimize_workers=True,
            )

            # Track which workers _optimize_workers would optimize
            # by checking the worker_training_examples passed to it
            original_optimize_workers = optimizer._optimize_workers

            def tracking_optimize_workers(*args, **kwargs):
                # Only return config for workers that have training data
                worker_examples = kwargs.get('worker_training_examples', args[0] if args else {})
                return {
                    name: {"prompts": {"role_definition": f"Optimized {name}"}}
                    for name in worker_examples.keys()
                    if name in mock_runner.workers
                }

            with patch.object(optimizer, '_optimize_workers', side_effect=tracking_optimize_workers) as mock_opt:
                result = optimizer.compile(
                    training_examples=manager_examples,
                    metric=mock_metric,
                    worker_training_examples=worker_training_examples,
                    optimizer="bootstrap",
                )

                # Only DataGatherer should be in config (only one with training data)
                assert "DataGatherer" in result.config.get("workers", {})
                # Summarizer should NOT be in workers (no training data provided)
                assert "Summarizer" not in result.config.get("workers", {})

    def test_compile_with_worker_specific_metrics(self):
        """Test compile() with per-worker custom metrics."""
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig

        mock_runner = self._create_mock_runner()

        manager_examples = [
            TrainingExample(
                inputs={"user_input": "Test"},
                expected_output="Result",
                full_trace="Trace..."
            ),
        ]

        worker_training_examples = {
            "DataGatherer": [
                TrainingExample(
                    inputs={"user_input": "search"},
                    expected_output="results",
                    full_trace="trace..."
                ),
            ],
        }

        def default_metric(ex, pred, trace=None):
            return True

        def gatherer_metric(ex, pred, trace=None):
            return "search" in str(getattr(pred, 'response', ''))

        worker_metrics = {
            "DataGatherer": gatherer_metric,
        }

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(
                mock_runner,
                optimize_workers=True,
            )

            # Mock _optimize_workers to verify correct metrics are passed
            captured_metrics = {}
            def capture_optimize_workers(worker_training_examples, worker_metrics, default_metric, **kwargs):
                # Check which metric would be used for each worker
                for worker_name in worker_training_examples:
                    if worker_metrics and worker_name in worker_metrics:
                        captured_metrics[worker_name] = worker_metrics[worker_name]
                    else:
                        captured_metrics[worker_name] = default_metric
                return {worker_name: {"prompts": {}} for worker_name in worker_training_examples}

            with patch.object(optimizer, '_optimize_workers', side_effect=capture_optimize_workers):
                result = optimizer.compile(
                    training_examples=manager_examples,
                    metric=default_metric,
                    worker_training_examples=worker_training_examples,
                    worker_metrics=worker_metrics,
                    optimizer="bootstrap",
                )

            # Verify DataGatherer would get its specific metric
            assert captured_metrics.get("DataGatherer") == gatherer_metric

    def test_compile_records_workers_optimized_in_provenance(self):
        """Test that provenance records which workers were optimized."""
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig

        mock_runner = self._create_mock_runner()

        manager_examples = [
            TrainingExample(
                inputs={"user_input": "Test"},
                expected_output="Result",
                full_trace="Trace..."
            ),
        ]

        worker_training_examples = {
            "DataGatherer": [
                TrainingExample(
                    inputs={"user_input": "search"},
                    expected_output="results",
                    full_trace="trace..."
                ),
            ],
        }

        def mock_metric(ex, pred, trace=None):
            return True

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(
                mock_runner,
                optimize_workers=True,
            )

            # Mock _optimize_workers to return a config for DataGatherer
            mock_worker_configs = {
                "DataGatherer": {"prompts": {"role_definition": "Optimized"}}
            }
            with patch.object(optimizer, '_optimize_workers', return_value=mock_worker_configs):
                result = optimizer.compile(
                    training_examples=manager_examples,
                    metric=mock_metric,
                    worker_training_examples=worker_training_examples,
                    optimizer="bootstrap",
                )

            # Check provenance
            assert result.optimization.optimized == True
            last_run = result.optimization.runs[-1]
            assert "workers_optimized" in last_run.optimizer_config
            assert "DataGatherer" in last_run.optimizer_config["workers_optimized"]


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
        # SimpleLLM uses system_prompt, not role_definition
        assert optimizer.config.prompts.get("system_prompt") == "You are a classifier."
    
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
        mock_agent.llm = Mock()

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
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})

        examples = [
            TrainingExample(
                inputs={"user_input": "What is 6*7?"},
                expected_output="42",
                full_trace="User: What is 6*7?\nAssistant: The answer is 42"
            ),
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
        """Test that optimization completes even when no examples pass."""
        from fair_prompt_optimizer.optimizers import AgentOptimizer
        from fair_prompt_optimizer.config import TrainingExample, OptimizedConfig

        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.llm = Mock()

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
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})

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

            optimizer = AgentOptimizer(mock_agent)
            result = optimizer.compile(
                training_examples=examples,
                metric=mock_metric,
                optimizer="bootstrap",
            )

            # Verify provenance was recorded even with no examples passing
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


class TestCombinePromptComponents:
    """Test combine_prompt_components utility."""

    def test_combine_role_only(self):
        from fair_prompt_optimizer.optimizers.base import combine_prompt_components

        result = combine_prompt_components("You are a helpful assistant.")

        assert "<ROLE_DEFINITION>" in result
        assert "</ROLE_DEFINITION>" in result
        assert "You are a helpful assistant." in result

    def test_combine_with_format_instructions(self):
        from fair_prompt_optimizer.optimizers.base import combine_prompt_components

        result = combine_prompt_components(
            "You are a helper.",
            format_instructions=["Be concise", "Show your work"],
        )

        assert "<ROLE_DEFINITION>" in result
        assert "<FORMAT_INSTRUCTIONS>" in result
        assert "<FORMAT_ITEM>" in result
        assert "Be concise" in result
        assert "Show your work" in result

    def test_combine_with_dict_format_instructions(self):
        from fair_prompt_optimizer.optimizers.base import combine_prompt_components

        result = combine_prompt_components(
            "Role",
            format_instructions=[
                {"text": "Format 1"},
                {"content": "Format 2"},
            ],
        )

        assert "Format 1" in result
        assert "Format 2" in result


class TestParseOptimizedPrompt:
    """Test parse_optimized_prompt utility."""

    def test_parse_valid_optimized_text(self):
        from fair_prompt_optimizer.optimizers.base import parse_optimized_prompt

        optimized_text = """
        <ROLE_DEFINITION>
        You are an optimized assistant.
        </ROLE_DEFINITION>

        <FORMAT_INSTRUCTIONS>
        <FORMAT_ITEM>
        Be very concise.
        </FORMAT_ITEM>
        <FORMAT_ITEM>
        Always explain.
        </FORMAT_ITEM>
        </FORMAT_INSTRUCTIONS>
        """

        result = parse_optimized_prompt(
            optimized_text,
            original_role="Original role",
            original_format_instructions=["Old format"],
        )

        assert result.role_definition == "You are an optimized assistant."
        assert result.role_definition_changed == True
        assert len(result.format_instructions) == 2
        assert "Be very concise." in result.format_instructions
        assert result.format_instructions_changed == True

    def test_parse_with_typo_in_closing_tag(self):
        from fair_prompt_optimizer.optimizers.base import parse_optimized_prompt

        # LLMs sometimes make typos in XML tags
        optimized_text = """
        <ROLE_DEFINITION>
        New role
        </ROLE_DEF>
        """

        result = parse_optimized_prompt(
            optimized_text,
            original_role="Original",
        )

        assert result.role_definition == "New role"
        assert result.role_definition_changed == True

    def test_parse_falls_back_to_original(self):
        from fair_prompt_optimizer.optimizers.base import parse_optimized_prompt

        # Text without proper markers
        optimized_text = "Just some random text without markers"

        result = parse_optimized_prompt(
            optimized_text,
            original_role="Original role",
            original_format_instructions=["Original format"],
        )

        assert result.role_definition == "Original role"
        assert result.role_definition_changed == False
        assert result.format_instructions == ["Original format"]
        assert result.format_instructions_changed == False

    def test_parse_unchanged_role(self):
        from fair_prompt_optimizer.optimizers.base import parse_optimized_prompt

        optimized_text = """
        <ROLE_DEFINITION>
        Same role
        </ROLE_DEFINITION>
        """

        result = parse_optimized_prompt(
            optimized_text,
            original_role="Same role",
        )

        assert result.role_definition_changed == False


class TestOptimizedPromptsDataclass:
    """Test the OptimizedPrompts dataclass."""

    def test_default_values(self):
        from fair_prompt_optimizer.config import OptimizedPrompts

        prompts = OptimizedPrompts()

        assert prompts.role_definition is None
        assert prompts.role_definition_changed == False
        assert prompts.format_instructions is None
        assert prompts.format_instructions_changed == False


class TestMultiAgentOptimizerTest:
    """Test MultiAgentOptimizer.test() method."""

    def test_multi_agent_test_method(self):
        from fair_prompt_optimizer.optimizers import MultiAgentOptimizer
        from unittest.mock import Mock, patch

        # Create mock runner
        mock_runner = Mock()

        async def mock_arun(input_text):
            return f"Response to: {input_text}"
        mock_runner.arun = mock_arun
        mock_runner.manager = Mock()
        mock_runner.manager.planner = Mock()
        mock_runner.manager.planner.prompt_builder = Mock()
        mock_runner.manager.planner.prompt_builder.role_definition = None
        mock_runner.workers = {}

        with patch('fair_prompt_optimizer.config.extract_multi_agent_config') as mock_extract:
            mock_extract.return_value = {
                "version": "1.0",
                "type": "multi_agent",
                "manager": {"prompts": {}},
                "workers": {},
            }

            optimizer = MultiAgentOptimizer(mock_runner)
            result = optimizer.test("Test input")

            assert "Test input" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])