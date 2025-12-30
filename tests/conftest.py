# conftest.py
"""
Shared fixtures and mocks for fair_prompt_optimizer tests.
"""

import pytest
import sys
from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional


# =============================================================================
# Mock fairlib Components
# =============================================================================

class MockRoleDefinition:
    """Mock for fairlib.RoleDefinition"""
    def __init__(self, text: str):
        self.text = text


class MockExample:
    """Mock for fairlib.Example"""
    def __init__(self, text: str):
        self.text = text


class MockWorkingMemory:
    """Mock for fairlib.WorkingMemory"""
    def __init__(self):
        self._memory = {}
    
    def clear(self):
        self._memory = {}
    
    def set(self, key: str, value: Any):
        self._memory[key] = value
    
    def get(self, key: str, default=None):
        return self._memory.get(key, default)


class MockTool:
    """Base mock tool class"""
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def execute(self, **kwargs) -> str:
        return "mock result"


class MockSafeCalculatorTool(MockTool):
    """Mock for fairlib.SafeCalculatorTool"""
    name = "calculator"
    description = "Performs safe mathematical calculations"
    
    def execute(self, expression: str) -> str:
        try:
            # Simple safe eval for basic math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {e}"


class MockToolRegistry:
    """Mock for fairlib.ToolRegistry"""
    def __init__(self):
        self._tools: Dict[str, MockTool] = {}
    
    def register_tool(self, tool: MockTool):
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[MockTool]:
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, MockTool]:
        return self._tools


class MockToolExecutor:
    """Mock for fairlib.ToolExecutor"""
    def __init__(self, registry: MockToolRegistry):
        self.registry = registry
    
    def execute(self, tool_name: str, **kwargs) -> str:
        tool = self.registry.get_tool(tool_name)
        if tool:
            return tool.execute(**kwargs)
        return f"Error: Tool {tool_name} not found"


class MockPromptBuilder:
    """Mock for the prompt builder component"""
    def __init__(self):
        self.role_definition = MockRoleDefinition("You are a helpful assistant.")
        self.examples = []


class MockPlanner:
    """Mock for fairlib.SimpleReActPlanner"""
    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry
        self.prompt_builder = MockPromptBuilder()
    
    async def plan(self, query: str) -> Dict[str, Any]:
        return {"action": "final_answer", "response": "Mock response"}


class MockLLMAdapter:
    """Mock for fairlib LLM adapters (HuggingFaceAdapter, etc.)"""
    def __init__(self, model_name: str = "mock-model", **kwargs):
        self.model_name = model_name
        self.model = model_name
        self._kwargs = kwargs
        self._responses = ["42", "The answer is 42"]
        self._response_idx = 0
    
    def generate(self, prompt: str) -> str:
        response = self._responses[self._response_idx % len(self._responses)]
        self._response_idx += 1
        return response
    
    async def agenerate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def set_responses(self, responses: List[str]):
        """Set the sequence of responses for testing"""
        self._responses = responses
        self._response_idx = 0


class MockAgent:
    """Mock for fairlib.SimpleAgent"""
    def __init__(
        self,
        llm: MockLLMAdapter,
        planner: MockPlanner,
        tool_executor: MockToolExecutor,
        memory: MockWorkingMemory,
        max_steps: int = 10
    ):
        self.llm = llm
        self.planner = planner
        self.tool_executor = tool_executor
        self.memory = memory
        self.max_steps = max_steps
        self._responses = ["42"]
        self._response_idx = 0
    
    def run(self, user_input: str) -> str:
        response = self._responses[self._response_idx % len(self._responses)]
        self._response_idx += 1
        return response
    
    async def arun(self, user_input: str) -> str:
        return self.run(user_input)
    
    def set_responses(self, responses: List[str]):
        """Set the sequence of responses for testing"""
        self._responses = responses
        self._response_idx = 0


# =============================================================================
# Mock fairlib Module
# =============================================================================

def create_mock_fairlib():
    """Create a mock fairlib module with all necessary components"""
    mock_fairlib = MagicMock()
    
    # Adapters
    mock_fairlib.HuggingFaceAdapter = MockLLMAdapter
    mock_fairlib.OpenAIAdapter = MockLLMAdapter
    mock_fairlib.AnthropicAdapter = MockLLMAdapter
    
    # Tools
    mock_fairlib.SafeCalculatorTool = MockSafeCalculatorTool
    
    # Components
    mock_fairlib.ToolRegistry = MockToolRegistry
    mock_fairlib.ToolExecutor = MockToolExecutor
    mock_fairlib.WorkingMemory = MockWorkingMemory
    mock_fairlib.RoleDefinition = MockRoleDefinition
    mock_fairlib.Example = MockExample
    
    # Planners
    mock_fairlib.SimpleReActPlanner = MockPlanner
    
    # Agents
    mock_fairlib.SimpleAgent = MockAgent
    
    return mock_fairlib


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_fairlib_module():
    """Fixture that patches fairlib with mocks"""
    # Clear existing registries before patching
    import fair_prompt_optimizer.registry as registry_module
    registry_module.ADAPTER_REGISTRY.clear()
    registry_module.TOOL_REGISTRY.clear()
    registry_module.PLANNER_REGISTRY.clear()
    registry_module.AGENT_REGISTRY.clear()
    
    mock_fairlib = create_mock_fairlib()
    sys.modules['fairlib'] = mock_fairlib
    
    # Re-populate registries with mocked fairlib
    registry_module._populate_registries()
    
    yield mock_fairlib
    
    # Cleanup
    if 'fairlib' in sys.modules:
        del sys.modules['fairlib']


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM adapter"""
    return MockLLMAdapter("test-model")


@pytest.fixture
def mock_tool_registry():
    """Fixture providing a mock tool registry with calculator"""
    registry = MockToolRegistry()
    registry.register_tool(MockSafeCalculatorTool())
    return registry


@pytest.fixture
def mock_memory():
    """Fixture providing mock working memory"""
    return MockWorkingMemory()


@pytest.fixture
def mock_planner(mock_llm, mock_tool_registry):
    """Fixture providing a mock planner"""
    return MockPlanner(mock_llm, mock_tool_registry)


@pytest.fixture
def mock_agent(mock_llm, mock_planner, mock_tool_registry, mock_memory):
    """Fixture providing a fully configured mock agent"""
    executor = MockToolExecutor(mock_tool_registry)
    agent = MockAgent(
        llm=mock_llm,
        planner=mock_planner,
        tool_executor=executor,
        memory=mock_memory,
        max_steps=10
    )
    return agent


@pytest.fixture
def sample_training_examples():
    """Fixture providing sample training examples"""
    from fair_prompt_optimizer.translator import TrainingExample
    return [
        TrainingExample(
            inputs={"user_input": "What is 15 + 27?"},
            expected_output="42"
        ),
        TrainingExample(
            inputs={"user_input": "Calculate 100 divided by 4"},
            expected_output="25"
        ),
        TrainingExample(
            inputs={"user_input": "What is 8 times 7?"},
            expected_output="56"
        ),
    ]


@pytest.fixture
def sample_fair_config():
    """Fixture providing a sample FAIRConfig"""
    from fair_prompt_optimizer.translator import (
        FAIRConfig, ModelConfig, AgentConfig, OptimizationMetadata
    )
    return FAIRConfig(
        version="1.0",
        role_definition="You are a helpful calculator assistant.",
        examples=["User: What is 2+2?\nResponse: 4"],
        model=ModelConfig(
            model_name="test-model",
            adapter="HuggingFaceAdapter",
            adapter_kwargs={}
        ),
        agent=AgentConfig(
            agent_type="SimpleAgent",
            planner_type="SimpleReActPlanner",
            max_steps=10,
            tools=["SafeCalculatorTool"]
        ),
        metadata=OptimizationMetadata(
            optimized=True,
            optimizer="bootstrap"
        )
    )


@pytest.fixture
def temp_config_file(tmp_path, sample_fair_config):
    """Fixture providing a temporary config file"""
    from fair_prompt_optimizer.translator import save_fair_config
    config_path = tmp_path / "test_config.json"
    save_fair_config(sample_fair_config, str(config_path))
    return config_path


@pytest.fixture
def temp_examples_file(tmp_path, sample_training_examples):
    """Fixture providing a temporary training examples file"""
    from fair_prompt_optimizer.translator import save_training_examples
    examples_path = tmp_path / "test_examples.json"
    save_training_examples(sample_training_examples, str(examples_path))
    return examples_path


# =============================================================================
# Mock DSPy Components
# =============================================================================

@pytest.fixture
def mock_dspy():
    """Fixture that provides mock DSPy components"""
    import dspy
    
    # Create a mock LM for DSPy
    mock_lm = MagicMock()
    mock_lm.return_value = "Mock response"
    
    return {
        "module": dspy,
        "mock_lm": mock_lm
    }