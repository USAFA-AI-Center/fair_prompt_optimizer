#!/usr/bin/env python3
"""
Test Config Manager Interface

Tests every function in fairlib's config_manager.py to verify the forward and
backward translation between fairlib objects and JSON configs works correctly.

This ensures the contract between fair_prompt_optimizer and fairlib is maintained.

Functions Tested:
-----------------
PromptBuilder Serialization:
  - extract_prompts(builder) -> dict
  - apply_prompts(prompts_dict, builder) -> PromptBuilder

Single Agent Config:
  - extract_config(agent) -> dict
  - save_agent_config(agent, path) -> dict
  - load_agent_config(path) -> dict
  - load_agent(path, llm) -> SimpleAgent
  - load_prompts_into_agent(path, agent) -> None

Multi-Agent Config:
  - extract_multi_agent_config(runner) -> dict
  - save_multi_agent_config(runner, path) -> dict
  - load_multi_agent(path, llm) -> HierarchicalAgentRunner
  - load_agent_from_config_dict(config, llm) -> SimpleAgent

Usage:
    pytest tests/test_config_manager_interface.py -v
    # Or run directly:
    python tests/test_config_manager_interface.py
"""

import json
import tempfile
from pathlib import Path

# Test result tracking
TESTS_PASSED = 0
TESTS_FAILED = 0


def check(name: str, condition: bool, details: str = ""):
    """Record a test result."""
    global TESTS_PASSED, TESTS_FAILED
    if condition:
        TESTS_PASSED += 1
        print(f"  [PASS] {name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}")
        if details:
            print(f"         {details}")


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subsection(title: str):
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


def test_config_manager_interface():
    """Run all config_manager interface tests."""

    print_section("CONFIG MANAGER INTERFACE TESTS")
    print("Testing forward/backward translation with fairlib")

    # =========================================================================
    # IMPORTS
    # =========================================================================
    print_subsection("Importing fairlib modules")

    try:
        from fairlib import (
            HuggingFaceAdapter,
            SimpleAgent,
            ToolExecutor,
            ToolRegistry,
            WorkingMemory,
        )
        from fairlib.core.prompts import (
            Example,
            FormatInstruction,
            PromptBuilder,
            RoleDefinition,
            ToolInstruction,
            WorkerInstruction,
        )
        from fairlib.modules.action.tools.builtin_tools.safe_calculator import SafeCalculatorTool
        from fairlib.modules.agent.multi_agent_runner import HierarchicalAgentRunner
        from fairlib.modules.planning.react_planner import ReActPlanner

        # Import config_manager functions
        from fairlib.utils.config_manager import (
            _get_tool_registry,
            apply_prompts,
            extract_config,
            extract_multi_agent_config,
            extract_prompts,
            load_agent,
            load_agent_config,
            load_agent_from_config_dict,
            load_multi_agent,
            load_prompts_into_agent,
            save_agent_config,
            save_multi_agent_config,
        )

        print("  All imports successful")
        check("fairlib imports", True)

    except ImportError as e:
        print(f"  Import error: {e}")
        print("  Make sure fairlib is installed: pip install -e ../fair_llm")
        check("fairlib imports", False, str(e))
        return

    # Create temp directory for test files
    temp_dir = Path(tempfile.mkdtemp(prefix="config_manager_test_"))
    print(f"  Using temp directory: {temp_dir}")

    # =========================================================================
    # TEST 1: PromptBuilder Serialization (extract_prompts / apply_prompts)
    # =========================================================================
    print_section("TEST 1: PromptBuilder Serialization")

    print_subsection("Creating PromptBuilder with all field types")

    # Create a PromptBuilder with all possible field types
    original_builder = PromptBuilder()
    original_builder.role_definition = RoleDefinition(
        "You are a helpful math assistant that uses tools to solve problems."
    )
    original_builder.tool_instructions = [
        ToolInstruction("safe_calculator", "Performs arithmetic calculations safely"),
        ToolInstruction("unit_converter", "Converts between different units"),
    ]
    original_builder.worker_instructions = [
        WorkerInstruction("DataGatherer", "Collects information from various sources"),
        WorkerInstruction("Analyzer", "Analyzes data and provides insights"),
    ]
    original_builder.format_instructions = [
        FormatInstruction("Always respond in JSON format with 'thought' and 'action' keys."),
        FormatInstruction("Use 'final_answer' tool when you have the solution."),
    ]
    original_builder.examples = [
        Example(
            'User: What is 2+2?\nAssistant: {"thought": "Simple addition", "action": {"tool_name": "final_answer", "tool_input": "4"}}'
        ),
        Example(
            'User: Convert 5 miles to km\nAssistant: {"thought": "Need conversion", "action": {"tool_name": "unit_converter", "tool_input": "5 miles to km"}}'
        ),
    ]

    print("  Original PromptBuilder created with:")
    print(f"    - role_definition: {original_builder.role_definition.text[:50]}...")
    print(f"    - tool_instructions: {len(original_builder.tool_instructions)} items")
    print(f"    - worker_instructions: {len(original_builder.worker_instructions)} items")
    print(f"    - format_instructions: {len(original_builder.format_instructions)} items")
    print(f"    - examples: {len(original_builder.examples)} items")

    print_subsection("Testing extract_prompts()")

    # Forward translation: PromptBuilder -> dict
    prompts_dict = extract_prompts(original_builder)

    check("extract_prompts returns dict", isinstance(prompts_dict, dict))
    check("role_definition extracted", prompts_dict.get("role_definition") is not None)
    check("tool_instructions extracted", len(prompts_dict.get("tool_instructions", [])) == 2)
    check("worker_instructions extracted", len(prompts_dict.get("worker_instructions", [])) == 2)
    check("format_instructions extracted", len(prompts_dict.get("format_instructions", [])) == 2)
    check("examples extracted", len(prompts_dict.get("examples", [])) == 2)

    # Check tool_instructions structure
    ti = prompts_dict["tool_instructions"][0]
    check("tool_instruction has name", "name" in ti)
    check("tool_instruction has description", "description" in ti)

    # Check worker_instructions structure
    wi = prompts_dict["worker_instructions"][0]
    check("worker_instruction has name", "name" in wi)
    check("worker_instruction has role_description", "role_description" in wi)

    print_subsection("Testing apply_prompts()")

    # Backward translation: dict -> PromptBuilder
    restored_builder = PromptBuilder()
    apply_prompts(prompts_dict, restored_builder)

    check(
        "role_definition restored",
        restored_builder.role_definition is not None
        and restored_builder.role_definition.text == original_builder.role_definition.text,
    )
    check(
        "tool_instructions count restored",
        len(restored_builder.tool_instructions) == len(original_builder.tool_instructions),
    )
    check(
        "worker_instructions count restored",
        len(restored_builder.worker_instructions) == len(original_builder.worker_instructions),
    )
    check(
        "format_instructions count restored",
        len(restored_builder.format_instructions) == len(original_builder.format_instructions),
    )
    check(
        "examples count restored", len(restored_builder.examples) == len(original_builder.examples)
    )

    # Verify content matches
    check(
        "tool_instructions[0].name matches",
        restored_builder.tool_instructions[0].name == original_builder.tool_instructions[0].name,
    )
    check(
        "worker_instructions[0].name matches",
        restored_builder.worker_instructions[0].name
        == original_builder.worker_instructions[0].name,
    )
    check(
        "format_instructions[0].text matches",
        restored_builder.format_instructions[0].text
        == original_builder.format_instructions[0].text,
    )
    check(
        "examples[0].text matches",
        restored_builder.examples[0].text == original_builder.examples[0].text,
    )

    print_subsection("Round-trip verification")

    # Extract again and compare
    restored_dict = extract_prompts(restored_builder)
    check(
        "round-trip role_definition matches",
        restored_dict["role_definition"] == prompts_dict["role_definition"],
    )
    check(
        "round-trip tool_instructions matches",
        restored_dict["tool_instructions"] == prompts_dict["tool_instructions"],
    )
    check(
        "round-trip worker_instructions matches",
        restored_dict["worker_instructions"] == prompts_dict["worker_instructions"],
    )
    check(
        "round-trip format_instructions matches",
        restored_dict["format_instructions"] == prompts_dict["format_instructions"],
    )
    check("round-trip examples matches", restored_dict["examples"] == prompts_dict["examples"])

    # =========================================================================
    # TEST 2: Single Agent Config
    # =========================================================================
    print_section("TEST 2: Single Agent Config")

    print_subsection("Creating test agent")

    # Create a simple LLM (mock or real depending on environment)
    try:
        llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
        print("  Using HuggingFaceAdapter")
    except Exception as e:
        print(f"  Note: Could not create HuggingFaceAdapter: {e}")
        print("  Some tests may be skipped")
        llm = None

    if llm:
        # Create tool registry with SafeCalculatorTool
        tool_registry = ToolRegistry()
        tool_registry.register_tool(SafeCalculatorTool())

        # Create PromptBuilder
        agent_builder = PromptBuilder()
        agent_builder.role_definition = RoleDefinition("You are a math agent.")
        agent_builder.format_instructions = [
            FormatInstruction("Respond with JSON containing 'thought' and 'action'."),
        ]
        agent_builder.examples = [
            Example('Example: 2+2 -> {"thought": "add", "action": {...}}'),
        ]

        # Create planner and agent
        planner = ReActPlanner(llm, tool_registry, prompt_builder=agent_builder)
        original_agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=ToolExecutor(tool_registry),
            memory=WorkingMemory(),
            max_steps=10,
            stateless=False,
        )

        print("  Agent created with SafeCalculatorTool")

        print_subsection("Testing extract_config()")

        config = extract_config(original_agent)

        check("extract_config returns dict", isinstance(config, dict))
        check("config has version", "version" in config)
        check("config has type", config.get("type") == "agent")
        check("config has prompts section", "prompts" in config)
        check("config has model section", "model" in config)
        check("config has agent section", "agent" in config)
        check("config has metadata section", "metadata" in config)

        # Check prompts section
        check(
            "prompts.role_definition present", config["prompts"].get("role_definition") is not None
        )
        check(
            "prompts.format_instructions present",
            len(config["prompts"].get("format_instructions", [])) > 0,
        )
        check("prompts.examples present", len(config["prompts"].get("examples", [])) > 0)

        # Check model section
        check("model.adapter present", "adapter" in config["model"])
        check("model.model_name present", "model_name" in config["model"])

        # Check agent section
        check("agent.agent_type present", config["agent"].get("agent_type") == "SimpleAgent")
        check(
            "agent.tools includes SafeCalculatorTool",
            "SafeCalculatorTool" in config["agent"].get("tools", []),
        )
        check("agent.max_steps present", config["agent"].get("max_steps") == 10)

        print_subsection("Testing save_agent_config() and load_agent_config()")

        config_path = temp_dir / "test_agent.json"
        saved_config = save_agent_config(original_agent, str(config_path))

        check("save_agent_config creates file", config_path.exists())
        check("save_agent_config returns dict", isinstance(saved_config, dict))

        # Load it back
        loaded_config = load_agent_config(str(config_path))

        check("load_agent_config returns dict", isinstance(loaded_config, dict))
        check("loaded config matches saved", loaded_config == saved_config)

        # Verify JSON is valid and readable
        with open(config_path) as f:
            json_content = json.load(f)
        check("saved file is valid JSON", json_content is not None)

        print_subsection("Testing load_agent()")

        loaded_agent = load_agent(str(config_path), llm)

        check("load_agent returns SimpleAgent", isinstance(loaded_agent, SimpleAgent))
        check("loaded agent has planner", hasattr(loaded_agent, "planner"))
        check("loaded agent has prompt_builder", hasattr(loaded_agent.planner, "prompt_builder"))

        # Verify prompts were loaded correctly
        loaded_prompts = extract_prompts(loaded_agent.planner.prompt_builder)
        check(
            "loaded role_definition matches",
            loaded_prompts["role_definition"] == config["prompts"]["role_definition"],
        )
        check(
            "loaded format_instructions match",
            loaded_prompts["format_instructions"] == config["prompts"]["format_instructions"],
        )

        print_subsection("Testing load_prompts_into_agent()")

        # Create a new config with different prompts
        modified_config_path = temp_dir / "modified_agent.json"
        modified_config = dict(loaded_config)
        modified_config["prompts"]["role_definition"] = "You are a MODIFIED math agent."
        modified_config["prompts"]["examples"] = ["MODIFIED example"]

        with open(modified_config_path, "w") as f:
            json.dump(modified_config, f)

        # Load modified prompts into existing agent
        load_prompts_into_agent(str(modified_config_path), loaded_agent)

        updated_prompts = extract_prompts(loaded_agent.planner.prompt_builder)
        check(
            "load_prompts_into_agent updates role_definition",
            updated_prompts["role_definition"] == "You are a MODIFIED math agent.",
        )
        check(
            "load_prompts_into_agent updates examples",
            "MODIFIED example" in updated_prompts["examples"],
        )

    else:
        print("  Skipping agent tests (no LLM available)")

    # =========================================================================
    # TEST 3: Multi-Agent Config
    # =========================================================================
    print_section("TEST 3: Multi-Agent Config")

    if llm:
        print_subsection("Creating multi-agent system")

        # Create Worker 1: DataGatherer
        gatherer_registry = ToolRegistry()
        gatherer_builder = PromptBuilder()
        gatherer_builder.role_definition = RoleDefinition("You gather data.")
        gatherer_builder.format_instructions = [FormatInstruction("Respond in JSON.")]

        gatherer_planner = ReActPlanner(llm, gatherer_registry, prompt_builder=gatherer_builder)
        data_gatherer = SimpleAgent(
            llm=llm,
            planner=gatherer_planner,
            tool_executor=ToolExecutor(gatherer_registry),
            memory=WorkingMemory(),
            max_steps=3,
        )

        # Create Worker 2: Summarizer
        summarizer_registry = ToolRegistry()
        summarizer_builder = PromptBuilder()
        summarizer_builder.role_definition = RoleDefinition("You summarize data.")
        summarizer_builder.format_instructions = [FormatInstruction("Respond in JSON.")]

        summarizer_planner = ReActPlanner(
            llm, summarizer_registry, prompt_builder=summarizer_builder
        )
        summarizer = SimpleAgent(
            llm=llm,
            planner=summarizer_planner,
            tool_executor=ToolExecutor(summarizer_registry),
            memory=WorkingMemory(),
            max_steps=3,
        )

        # Create Manager
        manager_registry = ToolRegistry()
        manager_builder = PromptBuilder()
        manager_builder.role_definition = RoleDefinition("You manage a research team.")
        manager_builder.worker_instructions = [
            WorkerInstruction("DataGatherer", "Gathers information"),
            WorkerInstruction("Summarizer", "Summarizes findings"),
        ]
        manager_builder.format_instructions = [FormatInstruction("Delegate using JSON.")]

        manager_planner = ReActPlanner(llm, manager_registry, prompt_builder=manager_builder)
        manager_agent = SimpleAgent(
            llm=llm,
            planner=manager_planner,
            tool_executor=ToolExecutor(manager_registry),
            memory=WorkingMemory(),
            max_steps=8,
        )

        # Create HierarchicalAgentRunner
        runner = HierarchicalAgentRunner(
            manager_agent=manager_agent,
            workers={
                "DataGatherer": data_gatherer,
                "Summarizer": summarizer,
            },
        )

        print("  Created HierarchicalAgentRunner with 2 workers")

        print_subsection("Testing extract_multi_agent_config()")

        multi_config = extract_multi_agent_config(runner)

        check("extract_multi_agent_config returns dict", isinstance(multi_config, dict))
        check("multi config has type=multi_agent", multi_config.get("type") == "multi_agent")
        check("multi config has manager section", "manager" in multi_config)
        check("multi config has workers section", "workers" in multi_config)
        check("workers contains DataGatherer", "DataGatherer" in multi_config.get("workers", {}))
        check("workers contains Summarizer", "Summarizer" in multi_config.get("workers", {}))
        check("multi config has max_delegation_steps", "max_delegation_steps" in multi_config)

        # Check manager config structure
        manager_config = multi_config["manager"]
        check("manager has prompts", "prompts" in manager_config)
        check(
            "manager has worker_instructions",
            len(manager_config["prompts"].get("worker_instructions", [])) == 2,
        )

        # Check worker config structure
        gatherer_config = multi_config["workers"]["DataGatherer"]
        check("worker has prompts", "prompts" in gatherer_config)
        check("worker has agent section", "agent" in gatherer_config)

        print_subsection("Testing save_multi_agent_config() and load")

        multi_config_path = temp_dir / "test_multi_agent.json"
        _saved_multi_config = save_multi_agent_config(runner, str(multi_config_path))

        check("save_multi_agent_config creates file", multi_config_path.exists())

        # Verify JSON structure
        with open(multi_config_path) as f:
            json_multi = json.load(f)
        check("saved multi config is valid JSON", json_multi is not None)
        check(
            "saved multi config preserves workers",
            set(json_multi["workers"].keys()) == {"DataGatherer", "Summarizer"},
        )

        print_subsection("Testing load_multi_agent()")

        loaded_runner = load_multi_agent(str(multi_config_path), llm)

        check(
            "load_multi_agent returns HierarchicalAgentRunner",
            isinstance(loaded_runner, HierarchicalAgentRunner),
        )
        check("loaded runner has manager", hasattr(loaded_runner, "manager"))
        check("loaded runner has workers", hasattr(loaded_runner, "workers"))
        check("loaded runner has correct worker count", len(loaded_runner.workers) == 2)
        check("loaded runner has DataGatherer worker", "DataGatherer" in loaded_runner.workers)
        check("loaded runner has Summarizer worker", "Summarizer" in loaded_runner.workers)

        # Verify manager prompts
        loaded_manager_prompts = extract_prompts(loaded_runner.manager.planner.prompt_builder)
        check(
            "loaded manager role_definition matches",
            loaded_manager_prompts["role_definition"]
            == manager_config["prompts"]["role_definition"],
        )

        print_subsection("Testing load_agent_from_config_dict()")

        # Test the internal helper function
        worker_config = multi_config["workers"]["DataGatherer"]
        rebuilt_worker = load_agent_from_config_dict(worker_config, llm)

        check(
            "load_agent_from_config_dict returns SimpleAgent",
            isinstance(rebuilt_worker, SimpleAgent),
        )

        rebuilt_prompts = extract_prompts(rebuilt_worker.planner.prompt_builder)
        check(
            "rebuilt worker has correct role_definition",
            rebuilt_prompts["role_definition"] == "You gather data.",
        )

    else:
        print("  Skipping multi-agent tests (no LLM available)")

    # =========================================================================
    # TEST 4: Tool Registry Helper
    # =========================================================================
    print_section("TEST 4: Tool Registry Helper")

    print_subsection("Testing _get_tool_registry()")

    tool_registry_map = _get_tool_registry()

    check("_get_tool_registry returns dict", isinstance(tool_registry_map, dict))
    check("SafeCalculatorTool in registry", "SafeCalculatorTool" in tool_registry_map)
    check("SafeCalculatorTool is a class", callable(tool_registry_map.get("SafeCalculatorTool")))

    # Instantiate a tool from the registry
    if "SafeCalculatorTool" in tool_registry_map:
        calc_tool = tool_registry_map["SafeCalculatorTool"]()
        check("Can instantiate tool from registry", calc_tool is not None)

    # =========================================================================
    # TEST 5: Edge Cases
    # =========================================================================
    print_section("TEST 5: Edge Cases")

    print_subsection("Empty PromptBuilder")

    empty_builder = PromptBuilder()
    empty_prompts = extract_prompts(empty_builder)

    check("extract_prompts handles empty builder", isinstance(empty_prompts, dict))
    check("empty builder has None role_definition", empty_prompts["role_definition"] is None)
    check("empty builder has empty tool_instructions", empty_prompts["tool_instructions"] == [])
    check("empty builder has empty examples", empty_prompts["examples"] == [])

    # Apply empty prompts to another builder
    # Note: apply_prompts only updates fields with truthy values
    # It clears lists (tool_instructions, worker_instructions, format_instructions, examples)
    # but does NOT clear role_definition if the new value is None/falsy
    another_builder = PromptBuilder()
    another_builder.role_definition = RoleDefinition("Will NOT be cleared")
    another_builder.examples = [Example("Will be cleared")]

    apply_prompts(empty_prompts, another_builder)
    check(
        "apply_prompts preserves role_definition when new value is None",
        another_builder.role_definition is not None
        and another_builder.role_definition.text == "Will NOT be cleared",
    )
    check("apply_prompts clears examples list", len(another_builder.examples) == 0)

    print_subsection("File not found handling")

    try:
        load_agent_config("/nonexistent/path/config.json")
        check("load_agent_config raises on missing file", False)
    except FileNotFoundError:
        check("load_agent_config raises on missing file", True)
    except Exception as e:
        check("load_agent_config raises on missing file", False, f"Wrong exception: {type(e)}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("TEST SUMMARY")

    total = TESTS_PASSED + TESTS_FAILED
    print(f"  Passed: {TESTS_PASSED}/{total}")
    print(f"  Failed: {TESTS_FAILED}/{total}")

    if TESTS_FAILED == 0:
        print()
        print("  All tests passed!")
    else:
        print()
        print(f"  {TESTS_FAILED} test(s) failed - review output above")

    # Cleanup
    print()
    print(f"  Temp files saved to: {temp_dir}")
    print("  (You can manually inspect these files)")

    print()
    print("=" * 60)
    print("CONFIG MANAGER INTERFACE TESTS COMPLETE")
    print("=" * 60)

    # For pytest: assert no failures
    assert TESTS_FAILED == 0, f"{TESTS_FAILED} test(s) failed"


if __name__ == "__main__":
    test_config_manager_interface()
