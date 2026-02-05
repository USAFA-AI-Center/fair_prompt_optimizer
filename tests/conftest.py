# tests/conftest.py
"""
Shared pytest fixtures and configuration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

# =============================================================================
# Skip markers for conditional tests
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (require LLM)")
    config.addinivalue_line("markers", "requires_fairlib: marks tests that require fairlib")


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Sample config dictionary."""
    return {
        "version": "1.0",
        "type": "agent",
        "prompts": {
            "role_definition": "You are a helpful math assistant.",
            "tool_instructions": [{"name": "calculator", "description": "Does math calculations"}],
            "worker_instructions": [],
            "format_instructions": ["Show your work", "Be concise"],
            "examples": ["Example 1", "Example 2"],
        },
        "model": {
            "adapter": "HuggingFaceAdapter",
            "model_name": "test-model",
            "adapter_kwargs": {},
        },
        "agent": {
            "planner_type": "SimpleReActPlanner",
            "tools": ["SafeCalculatorTool"],
            "max_steps": 10,
            "stateless": False,
        },
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create a sample config file."""
    path = temp_dir / "config.json"
    with open(path, "w") as f:
        json.dump(sample_config_dict, f, indent=2)
    return path


@pytest.fixture
def sample_training_data():
    """Sample training examples as list of dicts."""
    return [
        {"inputs": {"user_input": "What is 2+2?"}, "expected_output": "4"},
        {"inputs": {"user_input": "What is 10*5?"}, "expected_output": "50"},
        {"inputs": {"user_input": "What is 100/4?"}, "expected_output": "25"},
    ]


@pytest.fixture
def sample_training_file(temp_dir, sample_training_data):
    """Create a sample training data file."""
    path = temp_dir / "examples.json"
    with open(path, "w") as f:
        json.dump(sample_training_data, f, indent=2)
    return path


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = Mock()
    mock.__class__.__name__ = "MockAdapter"
    mock.model_name = "mock-model"
    mock.temperature = 0.7
    mock.max_tokens = 1024

    # Default response
    mock_response = Mock()
    mock_response.content = "Mock response"
    mock.invoke = Mock(return_value=mock_response)

    return mock


@pytest.fixture
def mock_prompt_builder():
    """Create a mock PromptBuilder."""
    mock = Mock()
    mock.role_definition = Mock()
    mock.role_definition.text = "Mock role"
    mock.tool_instructions = []
    mock.worker_instructions = []
    mock.format_instructions = []
    mock.examples = []
    return mock


@pytest.fixture
def mock_agent(mock_llm, mock_prompt_builder):
    """Create a mock agent."""
    mock_planner = Mock()
    mock_planner.prompt_builder = mock_prompt_builder
    mock_planner.tool_registry = Mock()
    mock_planner.tool_registry.get_all_tools = Mock(return_value={})
    mock_planner.__class__.__name__ = "MockPlanner"

    mock = Mock()
    mock.llm = mock_llm
    mock.planner = mock_planner
    mock.memory = Mock()
    mock.__class__.__name__ = "MockAgent"
    mock.max_steps = 10
    mock.stateless = False

    async def mock_arun(input_text):
        return f"Response to: {input_text}"

    mock.arun = mock_arun

    return mock


# =============================================================================
# Helper Functions
# =============================================================================


def make_mock_example(expected_output: str):
    """Create a mock example object."""
    mock = Mock()
    mock.expected_output = expected_output
    return mock


def make_mock_prediction(response: str):
    """Create a mock prediction object."""
    mock = Mock()
    mock.response = response
    return mock
