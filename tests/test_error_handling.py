# tests/test_error_handling.py
"""
Tests for error handling improvements in fair_prompt_optimizer.

Covers:
1. run_async() timeout behavior
2. Circuit breaker in _run_bootstrap()
3. TrainingExample.from_dict() input validation
4. forward() graceful error handling

Run with: pytest tests/test_error_handling.py -v
"""

import asyncio
import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from fair_prompt_optimizer.config import OptimizedConfig, TrainingExample
from fair_prompt_optimizer.optimizers.base import run_async


# =============================================================================
# 1. TestRunAsyncTimeout
# =============================================================================


class TestRunAsyncTimeout:
    """Test that run_async() with timeout properly times out."""

    def test_timeout_raises_on_slow_coroutine(self):
        """A coroutine that sleeps longer than the timeout should raise TimeoutError."""

        async def slow_coro():
            await asyncio.sleep(10)  # 10 seconds, well beyond our timeout
            return "should not reach here"

        with pytest.raises(asyncio.TimeoutError):
            run_async(slow_coro(), timeout=0.1)

    def test_fast_coroutine_completes_within_timeout(self):
        """A coroutine that finishes quickly should return normally."""

        async def fast_coro():
            await asyncio.sleep(0.01)
            return "done"

        result = run_async(fast_coro(), timeout=5)
        assert result == "done"

    def test_timeout_default_does_not_affect_fast_coroutine(self):
        """Default timeout (300s) should not interfere with a fast coroutine."""

        async def instant_coro():
            return 42

        result = run_async(instant_coro())
        assert result == 42

    def test_zero_timeout_raises_immediately(self):
        """A timeout of 0 should raise TimeoutError for any coroutine with work."""

        async def any_coro():
            await asyncio.sleep(0.01)
            return "nope"

        with pytest.raises(asyncio.TimeoutError):
            run_async(any_coro(), timeout=0)


# =============================================================================
# 2. TestCircuitBreaker
# =============================================================================


class TestCircuitBreaker:
    """Test bootstrap circuit breaker behavior in _run_bootstrap().

    Key architecture note: AgentModule.forward() catches agent exceptions
    internally and returns error strings. The circuit breaker in _run_bootstrap()
    catches exceptions that escape the module call (e.g., metric failures,
    memory reset errors, or unexpected exceptions). We test by patching
    AgentModule.__call__ to raise directly, simulating these escape scenarios.
    """

    def _create_mock_agent(self):
        """Helper to create a mock agent."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()

        async def noop_arun(user_input):
            return "ok"

        mock_agent.arun = noop_arun
        return mock_agent

    def _create_training_examples(self, count):
        """Helper to create N training examples."""
        return [
            TrainingExample(
                inputs={"user_input": f"Question {i}"},
                expected_output=f"Answer {i}",
                full_trace=f"User: Question {i}\nAssistant: Answer {i}",
            )
            for i in range(count)
        ]

    def test_stops_after_max_consecutive_failures(self):
        """Circuit breaker should stop bootstrap after 5 consecutive failures."""
        from fair_prompt_optimizer.optimizers.agent import AgentModule, AgentOptimizer

        call_count = 0
        original_call = AgentModule.__call__

        def failing_call(self, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Module crashed")

        mock_agent = self._create_mock_agent()
        examples = self._create_training_examples(10)

        def mock_metric(ex, pred, trace=None):
            return True

        with patch.object(OptimizedConfig, "from_agent") as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(
                config={
                    "version": "1.0",
                    "type": "agent",
                    "prompts": {"role_definition": "Test", "examples": []},
                }
            )
            optimizer = AgentOptimizer(mock_agent)

            with patch.object(AgentModule, "__call__", failing_call):
                result = optimizer._run_bootstrap(examples, mock_metric, max_demos=10)

        assert result == []
        assert call_count == 5  # Circuit breaker fires after 5 consecutive failures

    def test_partial_results_returned_before_circuit_breaker(self):
        """If some examples pass before consecutive failures start, partial results are returned."""
        from fair_prompt_optimizer.optimizers.agent import AgentModule, AgentOptimizer
        import dspy

        call_count = 0

        def pass_then_fail(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return dspy.Prediction(response=f"Good answer")
            raise RuntimeError("Module crashed")

        mock_agent = self._create_mock_agent()
        examples = self._create_training_examples(10)

        def always_passes_metric(ex, pred, trace=None):
            return True

        with patch.object(OptimizedConfig, "from_agent") as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(
                config={
                    "version": "1.0",
                    "type": "agent",
                    "prompts": {"role_definition": "Test", "examples": []},
                }
            )
            optimizer = AgentOptimizer(mock_agent)

            with patch.object(AgentModule, "__call__", pass_then_fail):
                result = optimizer._run_bootstrap(examples, always_passes_metric, max_demos=10)

        assert len(result) == 2  # First 2 succeeded with full_trace
        assert call_count == 7  # 2 successes + 5 failures before circuit breaker

    def test_consecutive_failures_reset_on_success(self):
        """A successful example should reset the consecutive failure counter."""
        from fair_prompt_optimizer.optimizers.agent import AgentModule, AgentOptimizer
        import dspy

        call_count = 0

        def intermittent_failures(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:
                return dspy.Prediction(response=f"Success")
            raise RuntimeError("Intermittent failure")

        mock_agent = self._create_mock_agent()
        examples = self._create_training_examples(10)

        def always_passes_metric(ex, pred, trace=None):
            return True

        with patch.object(OptimizedConfig, "from_agent") as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(
                config={
                    "version": "1.0",
                    "type": "agent",
                    "prompts": {"role_definition": "Test", "examples": []},
                }
            )
            optimizer = AgentOptimizer(mock_agent)

            with patch.object(AgentModule, "__call__", intermittent_failures):
                result = optimizer._run_bootstrap(examples, always_passes_metric, max_demos=10)

        # Calls 1-4 fail, call 5 succeeds (resets counter),
        # calls 6-9 fail, call 10 succeeds (resets counter)
        assert call_count == 10
        assert len(result) == 2  # 2 successes (call 5 and call 10)

    def test_circuit_breaker_logs_error(self, caplog):
        """Circuit breaker should log an error message when triggered."""
        from fair_prompt_optimizer.optimizers.agent import AgentModule, AgentOptimizer

        def always_fail(self, **kwargs):
            raise RuntimeError("Boom")

        mock_agent = self._create_mock_agent()
        examples = self._create_training_examples(6)

        def mock_metric(ex, pred, trace=None):
            return True

        with patch.object(OptimizedConfig, "from_agent") as mock_from_agent:
            mock_from_agent.return_value = OptimizedConfig(
                config={
                    "version": "1.0",
                    "type": "agent",
                    "prompts": {"role_definition": "Test", "examples": []},
                }
            )
            optimizer = AgentOptimizer(mock_agent)

            with caplog.at_level(logging.ERROR, logger="fair_prompt_optimizer.optimizers.agent"):
                with patch.object(AgentModule, "__call__", always_fail):
                    optimizer._run_bootstrap(examples, mock_metric, max_demos=10)

        assert any("Circuit breaker" in record.message for record in caplog.records)


# =============================================================================
# 3. TestTrainingExampleValidation
# =============================================================================


class TestTrainingExampleValidation:
    """Test input validation in TrainingExample.from_dict()."""

    def test_inputs_not_a_dict_raises_valueerror(self):
        """If 'inputs' is not a dict, from_dict() should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            TrainingExample.from_dict({
                "inputs": "not a dict",
                "expected_output": "some output",
            })

    def test_inputs_as_list_raises_valueerror(self):
        """If 'inputs' is a list instead of dict, from_dict() should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            TrainingExample.from_dict({
                "inputs": ["item1", "item2"],
                "expected_output": "some output",
            })

    def test_inputs_as_int_raises_valueerror(self):
        """If 'inputs' is an integer, from_dict() should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            TrainingExample.from_dict({
                "inputs": 42,
                "expected_output": "some output",
            })

    def test_empty_expected_output_raises_valueerror(self):
        """An empty string for expected_output should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            TrainingExample.from_dict({
                "inputs": {"user_input": "Hello"},
                "expected_output": "",
            })

    def test_whitespace_only_expected_output_raises_valueerror(self):
        """A whitespace-only expected_output should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            TrainingExample.from_dict({
                "inputs": {"user_input": "Hello"},
                "expected_output": "   ",
            })

    def test_missing_expected_output_raises_valueerror(self):
        """Missing expected_output key should raise ValueError (defaults to empty string)."""
        with pytest.raises(ValueError, match="non-empty string"):
            TrainingExample.from_dict({
                "inputs": {"user_input": "Hello"},
            })

    def test_non_string_expected_output_raises_valueerror(self):
        """A non-string expected_output should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            TrainingExample.from_dict({
                "inputs": {"user_input": "Hello"},
                "expected_output": 123,
            })

    def test_valid_data_succeeds(self):
        """Valid data should create a TrainingExample successfully."""
        example = TrainingExample.from_dict({
            "inputs": {"user_input": "What is 2+2?"},
            "expected_output": "4",
        })

        assert example.inputs == {"user_input": "What is 2+2?"}
        assert example.expected_output == "4"
        assert example.full_trace is None

    def test_valid_data_with_full_trace_succeeds(self):
        """Valid data including full_trace should create a TrainingExample successfully."""
        example = TrainingExample.from_dict({
            "inputs": {"user_input": "What is 2+2?"},
            "expected_output": "4",
            "full_trace": "User: What is 2+2?\nAssistant: 4",
        })

        assert example.inputs == {"user_input": "What is 2+2?"}
        assert example.expected_output == "4"
        assert example.full_trace == "User: What is 2+2?\nAssistant: 4"

    def test_missing_inputs_defaults_to_empty_dict(self):
        """Missing 'inputs' key should default to an empty dict (which is still a dict)."""
        example = TrainingExample.from_dict({
            "expected_output": "some output",
        })

        assert example.inputs == {}
        assert example.expected_output == "some output"


# =============================================================================
# 4. TestForwardErrorHandling
# =============================================================================


class TestForwardErrorHandling:
    """Test that forward() logs and handles errors gracefully."""

    def _create_agent_module(self, arun_fn):
        """Helper to create an AgentModule with a given arun function."""
        from fair_prompt_optimizer.optimizers.agent import AgentModule

        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.planner = Mock()
        mock_agent.planner.prompt_builder = Mock()
        mock_agent.planner.prompt_builder.role_definition = None
        mock_agent.planner.prompt_builder.format_instructions = []
        mock_agent.planner.tool_registry = Mock()
        mock_agent.planner.tool_registry.get_all_tools = Mock(return_value={})
        mock_agent.llm = Mock()
        mock_agent.arun = arun_fn

        return AgentModule(mock_agent)

    def test_runtime_error_returns_error_string(self):
        """forward() should catch RuntimeError and return an error string, not crash."""

        async def failing_arun(user_input):
            raise RuntimeError("LLM connection failed")

        module = self._create_agent_module(failing_arun)
        result = module(user_input="Hello")

        assert hasattr(result, "response")
        assert "Error" in result.response
        assert "LLM connection failed" in result.response

    def test_value_error_returns_error_string(self):
        """forward() should catch ValueError and return an error string."""

        async def bad_value_arun(user_input):
            raise ValueError("Invalid input format")

        module = self._create_agent_module(bad_value_arun)
        result = module(user_input="bad input")

        assert hasattr(result, "response")
        assert "Error" in result.response
        assert "Invalid input format" in result.response

    def test_generic_exception_returns_error_string(self):
        """forward() should catch any Exception and return an error string."""

        async def generic_fail_arun(user_input):
            raise Exception("Something unexpected happened")

        module = self._create_agent_module(generic_fail_arun)
        result = module(user_input="test")

        assert hasattr(result, "response")
        assert "Error" in result.response
        assert "Something unexpected happened" in result.response

    def test_forward_error_logs_message(self, caplog):
        """forward() should log the error when the agent fails."""

        async def failing_arun(user_input):
            raise RuntimeError("Agent exploded")

        module = self._create_agent_module(failing_arun)

        with caplog.at_level(logging.ERROR, logger="fair_prompt_optimizer.optimizers.agent"):
            module(user_input="Hello")

        assert any("Agent error during forward pass" in record.message for record in caplog.records)
        assert any("Agent exploded" in record.message for record in caplog.records)

    def test_forward_returns_prediction_on_success(self):
        """forward() should return a proper dspy.Prediction on success."""
        import dspy

        async def good_arun(user_input):
            return f"Response to: {user_input}"

        module = self._create_agent_module(good_arun)
        result = module(user_input="Hello world")

        assert isinstance(result, dspy.Prediction)
        assert result.response == "Response to: Hello world"

    def test_forward_returns_prediction_on_error(self):
        """forward() should still return a dspy.Prediction even on error."""
        import dspy

        async def failing_arun(user_input):
            raise RuntimeError("Oops")

        module = self._create_agent_module(failing_arun)
        result = module(user_input="Hello")

        # Even on error, it should be a Prediction, not an exception
        assert isinstance(result, dspy.Prediction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
