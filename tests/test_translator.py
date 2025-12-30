# test_translator.py
"""
Tests for fair_prompt_optimizer.translator module.
"""

import json

from fair_prompt_optimizer.translator import (
    TrainingExample,
    ToolInstruction,
    WorkerInstruction,
    OptimizationMetadata,
    ModelConfig,
    AgentConfig,
    FAIRConfig,
    DSPyTranslator,
    load_fair_config,
    save_fair_config,
    load_training_examples,
    save_training_examples,
)


class TestTrainingExample:
    """Tests for TrainingExample dataclass"""
    
    def test_basic_creation(self):
        """Test basic TrainingExample creation"""
        example = TrainingExample(
            inputs={"user_input": "What is 2+2?"},
            expected_output="4"
        )
        assert example.inputs == {"user_input": "What is 2+2?"}
        assert example.expected_output == "4"
        assert example.metadata is None
    
    def test_creation_with_metadata(self):
        """Test TrainingExample creation with metadata"""
        example = TrainingExample(
            inputs={"user_input": "Hello"},
            expected_output="Hi there!",
            metadata={"source": "test", "difficulty": "easy"}
        )
        assert example.metadata["source"] == "test"
        assert example.metadata["difficulty"] == "easy"
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        example = TrainingExample(
            inputs={"user_input": "Test"},
            expected_output="Result",
            metadata={"key": "value"}
        )
        d = example.to_dict()
        
        assert d["inputs"] == {"user_input": "Test"}
        assert d["expected_output"] == "Result"
        assert d["metadata"] == {"key": "value"}
    
    def test_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "inputs": {"user_input": "Question"},
            "expected_output": "Answer",
            "metadata": {"id": 1}
        }
        example = TrainingExample.from_dict(data)
        
        assert example.inputs == {"user_input": "Question"}
        assert example.expected_output == "Answer"
        assert example.metadata == {"id": 1}
    
    def test_roundtrip(self):
        """Test serialization roundtrip"""
        original = TrainingExample(
            inputs={"user_input": "Input", "context": "Context"},
            expected_output="Output",
            metadata={"score": 0.95}
        )
        
        restored = TrainingExample.from_dict(original.to_dict())
        
        assert restored.inputs == original.inputs
        assert restored.expected_output == original.expected_output
        assert restored.metadata == original.metadata
    
    def test_multiple_inputs(self):
        """Test TrainingExample with multiple input fields"""
        example = TrainingExample(
            inputs={
                "user_input": "What is the capital?",
                "context": "We are discussing France.",
                "language": "English"
            },
            expected_output="Paris"
        )
        
        assert len(example.inputs) == 3
        assert example.inputs["context"] == "We are discussing France."


class TestToolInstruction:
    """Tests for ToolInstruction dataclass"""
    
    def test_creation(self):
        """Test basic creation"""
        tool = ToolInstruction(
            name="calculator",
            description="Performs math operations"
        )
        assert tool.name == "calculator"
        assert tool.description == "Performs math operations"
    
    def test_roundtrip(self):
        """Test serialization roundtrip"""
        original = ToolInstruction(name="search", description="Web search")
        restored = ToolInstruction.from_dict(original.to_dict())
        
        assert restored.name == original.name
        assert restored.description == original.description


class TestWorkerInstruction:
    """Tests for WorkerInstruction dataclass"""
    
    def test_creation(self):
        """Test basic creation"""
        worker = WorkerInstruction(
            name="researcher",
            role_description="Researches topics deeply"
        )
        assert worker.name == "researcher"
        assert worker.role_description == "Researches topics deeply"
    
    def test_roundtrip(self):
        """Test serialization roundtrip"""
        original = WorkerInstruction(
            name="analyst",
            role_description="Analyzes data"
        )
        restored = WorkerInstruction.from_dict(original.to_dict())
        
        assert restored.name == original.name
        assert restored.role_description == original.role_description


class TestOptimizationMetadata:
    """Tests for OptimizationMetadata dataclass"""
    
    def test_default_creation(self):
        """Test creation with defaults"""
        meta = OptimizationMetadata()
        assert meta.optimized is False
        assert meta.optimized_at is None
        assert meta.optimizer is None
    
    def test_full_creation(self):
        """Test creation with all fields"""
        meta = OptimizationMetadata(
            optimized=True,
            optimized_at="2025-01-01T00:00:00",
            optimizer="bootstrap",
            optimizer_config={"max_demos": 4},
            score=0.95,
            num_training_examples=100
        )
        
        assert meta.optimized is True
        assert meta.optimizer == "bootstrap"
        assert meta.score == 0.95
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values"""
        meta = OptimizationMetadata(optimized=True, optimizer="mipro")
        d = meta.to_dict()
        
        assert "optimized" in d
        assert "optimizer" in d
        assert "optimized_at" not in d  # None values excluded
        assert "score" not in d


class TestModelConfig:
    """Tests for ModelConfig dataclass"""
    
    def test_default_creation(self):
        """Test creation with defaults"""
        config = ModelConfig()
        assert config.model_name == "dolphin3-qwen25-3b"
        assert config.adapter == "HuggingFaceAdapter"
        assert config.adapter_kwargs == {}
    
    def test_custom_creation(self):
        """Test creation with custom values"""
        config = ModelConfig(
            model_name="gpt-4",
            adapter="OpenAIAdapter",
            adapter_kwargs={"temperature": 0.7}
        )
        
        assert config.model_name == "gpt-4"
        assert config.adapter == "OpenAIAdapter"
        assert config.adapter_kwargs["temperature"] == 0.7
    
    def test_roundtrip(self):
        """Test serialization roundtrip"""
        original = ModelConfig(
            model_name="test-model",
            adapter="TestAdapter",
            adapter_kwargs={"key": "value"}
        )
        restored = ModelConfig.from_dict(original.to_dict())
        
        assert restored.model_name == original.model_name
        assert restored.adapter == original.adapter
        assert restored.adapter_kwargs == original.adapter_kwargs


class TestAgentConfig:
    """Tests for AgentConfig dataclass"""
    
    def test_default_creation(self):
        """Test creation with defaults"""
        config = AgentConfig()
        assert config.agent_type == "SimpleAgent"
        assert config.planner_type == "SimpleReActPlanner"
        assert config.max_steps == 10
        assert config.tools == []
    
    def test_custom_creation(self):
        """Test creation with custom values"""
        config = AgentConfig(
            agent_type="MultiAgent",
            planner_type="TreePlanner",
            max_steps=20,
            tools=["SafeCalculatorTool", "WebSearchTool"]
        )
        
        assert config.agent_type == "MultiAgent"
        assert config.max_steps == 20
        assert len(config.tools) == 2
    
    def test_roundtrip(self):
        """Test serialization roundtrip"""
        original = AgentConfig(
            agent_type="TestAgent",
            tools=["Tool1", "Tool2"]
        )
        restored = AgentConfig.from_dict(original.to_dict())
        
        assert restored.agent_type == original.agent_type
        assert restored.tools == original.tools


class TestFAIRConfig:
    """Tests for FAIRConfig dataclass"""
    
    def test_default_creation(self):
        """Test creation with defaults"""
        config = FAIRConfig()
        assert config.version == "1.0"
        assert config.role_definition is None
        assert config.examples == []
    
    def test_full_creation(self):
        """Test creation with all fields"""
        config = FAIRConfig(
            version="1.0",
            role_definition="You are a helpful assistant.",
            tool_instructions=[
                ToolInstruction(name="calc", description="Calculator")
            ],
            worker_instructions=[
                WorkerInstruction(name="helper", role_description="Helps")
            ],
            format_instructions=["Be concise"],
            examples=["Example 1", "Example 2"],
            model=ModelConfig(model_name="test-model"),
            agent=AgentConfig(max_steps=5),
            metadata=OptimizationMetadata(optimized=True)
        )
        
        assert config.role_definition == "You are a helpful assistant."
        assert len(config.tool_instructions) == 1
        assert len(config.examples) == 2
        assert config.metadata.optimized is True
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        config = FAIRConfig(
            role_definition="Test role",
            examples=["Ex1"],
            model=ModelConfig(model_name="model"),
            agent=AgentConfig(tools=["Tool1"])
        )
        
        d = config.to_dict()
        
        assert d["version"] == "1.0"
        assert d["role_definition"] == "Test role"
        assert d["examples"] == ["Ex1"]
        assert d["model"]["model_name"] == "model"
        assert d["agent"]["tools"] == ["Tool1"]
    
    def test_roundtrip(self):
        """Test full serialization roundtrip"""
        original = FAIRConfig(
            version="1.0",
            role_definition="Test assistant",
            tool_instructions=[
                ToolInstruction(name="tool1", description="desc1")
            ],
            examples=["User: Hi\nResponse: Hello"],
            model=ModelConfig(
                model_name="test-model",
                adapter="TestAdapter"
            ),
            agent=AgentConfig(
                agent_type="TestAgent",
                tools=["Tool1"]
            ),
            metadata=OptimizationMetadata(
                optimized=True,
                optimizer="bootstrap"
            )
        )
        
        restored = FAIRConfig.from_dict(original.to_dict())
        
        assert restored.version == original.version
        assert restored.role_definition == original.role_definition
        assert len(restored.tool_instructions) == len(original.tool_instructions)
        assert restored.examples == original.examples
        assert restored.model.model_name == original.model.model_name
        assert restored.agent.tools == original.agent.tools
        assert restored.metadata.optimized == original.metadata.optimized


class TestDSPyTranslator:
    """Tests for DSPyTranslator class"""
    
    def test_default_fields(self):
        """Test default field names"""
        translator = DSPyTranslator()
        assert translator.input_field == "user_input"
        assert translator.output_field == "response"
    
    def test_custom_fields(self):
        """Test custom field names"""
        translator = DSPyTranslator(
            input_field="query",
            output_field="answer"
        )
        assert translator.input_field == "query"
        assert translator.output_field == "answer"
    
    def test_training_examples_to_dspy(self):
        """Test conversion of TrainingExamples to DSPy Examples"""
        translator = DSPyTranslator()
        
        examples = [
            TrainingExample(
                inputs={"user_input": "Question 1"},
                expected_output="Answer 1"
            ),
            TrainingExample(
                inputs={"user_input": "Question 2"},
                expected_output="Answer 2"
            ),
        ]
        
        dspy_examples = translator.training_examples_to_dspy(examples)
        
        assert len(dspy_examples) == 2
        assert dspy_examples[0].user_input == "Question 1"
        assert dspy_examples[0].response == "Answer 1"
    
    def test_training_examples_fallback_fields(self):
        """Test fallback to alternative field names"""
        translator = DSPyTranslator()
        
        # Use a different field name than user_input
        examples = [
            TrainingExample(
                inputs={"query": "Test query"},
                expected_output="Test output"
            ),
        ]
        
        dspy_examples = translator.training_examples_to_dspy(examples)
        
        # Should fall back to first input value
        assert len(dspy_examples) == 1
        assert dspy_examples[0].user_input == "Test query"
    
    def test_parse_example_text(self):
        """Test parsing of example text strings"""
        translator = DSPyTranslator()
        
        text = "User: What is 2+2?\nThought: I need to calculate.\nResponse: 4"
        parsed = translator._parse_example_text(text)
        
        assert parsed["user_input"] == "What is 2+2?"
        assert parsed["reasoning"] == "I need to calculate."
        assert parsed["response"] == "4"
    
    def test_parse_example_text_multiline(self):
        """Test parsing multi-line example text"""
        translator = DSPyTranslator()
        
        text = """User: Tell me a story
Thought: I should create something engaging.
This requires creativity.
Response: Once upon a time..."""
        
        parsed = translator._parse_example_text(text)
        
        assert "Tell me a story" in parsed["user_input"]
        assert "creativity" in parsed["reasoning"]
    
    def test_demo_to_example_text(self):
        """Test conversion of demo dict to example text"""
        translator = DSPyTranslator()
        
        demo = {
            "user_input": "Hello",
            "reasoning": "Greet the user",
            "response": "Hi there!"
        }
        
        text = translator._demo_to_example_text(demo)
        
        assert "User: Hello" in text
        assert "Thought: Greet the user" in text
        assert "Response: Hi there!" in text


class TestFileIO:
    """Tests for file I/O functions"""
    
    def test_save_and_load_fair_config(self, tmp_path):
        """Test saving and loading FAIRConfig"""
        config = FAIRConfig(
            role_definition="Test role",
            model=ModelConfig(model_name="test-model"),
            agent=AgentConfig(tools=["Tool1"])
        )
        
        filepath = tmp_path / "config.json"
        save_fair_config(config, str(filepath))
        
        assert filepath.exists()
        
        loaded = load_fair_config(str(filepath))
        
        assert loaded.role_definition == config.role_definition
        assert loaded.model.model_name == config.model.model_name
        assert loaded.agent.tools == config.agent.tools
    
    def test_save_creates_directories(self, tmp_path):
        """Test that save creates parent directories"""
        config = FAIRConfig()
        filepath = tmp_path / "nested" / "dirs" / "config.json"
        
        save_fair_config(config, str(filepath))
        
        assert filepath.exists()
    
    def test_save_and_load_training_examples(self, tmp_path):
        """Test saving and loading training examples"""
        examples = [
            TrainingExample(
                inputs={"user_input": "Q1"},
                expected_output="A1"
            ),
            TrainingExample(
                inputs={"user_input": "Q2"},
                expected_output="A2",
                metadata={"id": 2}
            ),
        ]
        
        filepath = tmp_path / "examples.json"
        save_training_examples(examples, str(filepath))
        
        assert filepath.exists()
        
        loaded = load_training_examples(str(filepath))
        
        assert len(loaded) == 2
        assert loaded[0].inputs["user_input"] == "Q1"
        assert loaded[1].metadata["id"] == 2
    
    def test_config_json_format(self, tmp_path):
        """Test that saved JSON is properly formatted"""
        config = FAIRConfig(role_definition="Test")
        filepath = tmp_path / "config.json"
        
        save_fair_config(config, str(filepath))
        
        with open(filepath) as f:
            content = f.read()
        
        # Should be indented (pretty-printed)
        assert "\n" in content
        assert "  " in content  # Indentation
        
        # Should be valid JSON
        data = json.loads(content)
        assert data["role_definition"] == "Test"


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_training_example(self):
        """Test TrainingExample with empty inputs"""
        example = TrainingExample(inputs={}, expected_output="")
        assert example.inputs == {}
        assert example.expected_output == ""
    
    def test_unicode_content(self):
        """Test handling of unicode content"""
        example = TrainingExample(
            inputs={"user_input": "What is π × 2?"},
            expected_output="≈6.28"
        )
        
        restored = TrainingExample.from_dict(example.to_dict())
        assert "π" in restored.inputs["user_input"]
        assert "≈" in restored.expected_output
    
    def test_special_characters_in_examples(self):
        """Test handling of special characters"""
        config = FAIRConfig(
            examples=["User: Code:\n```python\nprint('hello')\n```\nResponse: Done"]
        )
        
        restored = FAIRConfig.from_dict(config.to_dict())
        assert "```python" in restored.examples[0]
    
    def test_very_long_content(self):
        """Test handling of very long content"""
        long_text = "x" * 10000
        example = TrainingExample(
            inputs={"user_input": long_text},
            expected_output=long_text
        )
        
        restored = TrainingExample.from_dict(example.to_dict())
        assert len(restored.inputs["user_input"]) == 10000