# Configuration Reference

Complete documentation for all configuration structures, CLI options, metrics, and APIs in `fair_prompt_optimizer`.

## Table of Contents

1. [Overview](#overview)
2. [Config Types](#config-types)
3. [SimpleLLM Config](#simplellm-config-type-simple_llm)
4. [Agent Config](#agent-config-type-agent)
5. [MultiAgent Config](#multiagent-config-type-multi_agent)
6. [Training Examples](#training-examples)
7. [The full_trace Format](#the-full_trace-format)
8. [Optimization Provenance](#optimization-provenance)
9. [CLI Reference](#cli-reference)
10. [Metrics Reference](#metrics-reference)
11. [Python API Reference](#python-api-reference)
12. [Complete Examples](#complete-examples)

---

## Overview

All configs share a common structure with type-specific sections:

```
+---------------------------------------------------------------------+
|                      CONFIG STRUCTURE                               |
+---------------------------------------------------------------------+
|  version        |  Config format version (e.g., "1.0")              |
|  type           |  "simple_llm" | "agent" | "multi_agent"           |
|  prompts        |  Prompt components (role, examples, etc.)         |
|  model          |  LLM adapter configuration                        |
|  agent          |  Agent-specific settings (if applicable)          |
|  optimization   |  Provenance tracking (added after optimize)       |
+---------------------------------------------------------------------+
```

### Config Lifecycle

```
+----------------+    optimize()    +---------------------+
| Initial        | --------------> | Optimized           |
| Config         |                 | Config              |
+----------------+                 +---------------------+
| - prompts      |                 | - prompts           |
|   - examples   |                 |   - examples <----  | (populated few-shot examples)
|     = []       |                 |     = [...]         |
|                |                 | - optimization      |
|                |                 |   - runs: [...]     |
+----------------+                 +---------------------+
```

### Compatibility

- **fairlib compatible**: Configs work with fairlib's `load_agent()` and `load_multi_agent()` functions
- **Backward compatible**: fairlib can load optimized configs directly (ignores `optimization` section)
- **Iterative optimization**: Multiple optimization runs are recorded in provenance

---

## Config Types

| Type | Optimizer Class | Use Case | Requires `full_trace` |
|------|-----------------|----------|----------------------|
| `simple_llm` | `SimpleLLMOptimizer` | Classification, format compliance, simple generation | No |
| `agent` | `AgentOptimizer` | Tool-using agents, ReAct workflows | Yes (recommended) |
| `multi_agent` | `MultiAgentOptimizer` | Manager + worker hierarchies | Yes (required) |

---

## SimpleLLM Config (type: "simple_llm")

For optimizing a raw LLM with a system prompt - no agent pipeline or tools.

### Schema

```json
{
  "version": "1.0",
  "type": "simple_llm",

  "prompts": {
    "role_definition": "<string>",
    "tool_instructions": [],
    "worker_instructions": [],
    "format_instructions": ["<string>", ...],
    "examples": ["<string>", ...]
  },

  "model": {
    "adapter": "<string>",
    "model_name": "<string>",
    "adapter_kwargs": {}
  },

  "agent": {
    "planner_type": "None",
    "tools": [],
    "max_steps": 1,
    "stateless": true
  },

  "optimization": {
    "runs": [<OptimizationRun>, ...]
  }
}
```

### Field Reference

#### `prompts`

| Field | Type | Description |
|-------|------|-------------|
| `role_definition` | `string` | The system prompt/instructions for the LLM. This is what gets optimized by MIPROv2. |
| `format_instructions` | `string[]` | Output format requirements. Also optimized by MIPROv2. |
| `examples` | `string[]` | Few-shot examples added by optimization. Each is a formatted input/output pair. |
| `tool_instructions` | `array` | Empty for simple_llm. |
| `worker_instructions` | `array` | Empty for simple_llm. |

#### `model`

| Field | Type | Description |
|-------|------|-------------|
| `adapter` | `string` | LLM adapter class name. One of: `HuggingFaceAdapter`, `OpenAIAdapter`, `OllamaAdapter` |
| `model_name` | `string` | Model identifier (e.g., `"dolphin3-qwen25-3b"`, `"gpt-4o"`, `"llama3:8b"`) |
| `adapter_kwargs` | `object` | Additional kwargs passed to adapter constructor (e.g., `{"temperature": 0.7}`) |

#### `agent`

For `simple_llm`, this section is minimal:

| Field | Type | Description |
|-------|------|-------------|
| `planner_type` | `string` | Always `"None"` for simple_llm |
| `tools` | `array` | Always empty `[]` |
| `max_steps` | `int` | Always `1` |
| `stateless` | `bool` | Always `true` |

### Example: Sentiment Classifier

```json
{
  "version": "1.0",
  "type": "simple_llm",

  "prompts": {
    "role_definition": "You are a sentiment classifier.\n\n# RESPONSE FORMAT (MANDATORY)\nRespond with ONLY one of:\nSENTIMENT: positive\nSENTIMENT: negative\nSENTIMENT: neutral\n\nNo explanations, just the label.",
    "format_instructions": [],
    "examples": []
  },

  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },

  "agent": {
    "planner_type": "None",
    "tools": [],
    "max_steps": 1,
    "stateless": true
  }
}
```

---

## Agent Config (type: "agent")

For optimizing a fairlib `SimpleAgent` with tools and ReAct planning.

### Schema

```json
{
  "version": "1.0",
  "type": "agent",

  "prompts": {
    "role_definition": "<string>",
    "tool_instructions": [<ToolInstruction>, ...],
    "worker_instructions": [<WorkerInstruction>, ...],
    "format_instructions": ["<string>", ...],
    "examples": ["<string>", ...]
  },

  "model": {
    "adapter": "<string>",
    "model_name": "<string>",
    "adapter_kwargs": {}
  },

  "agent": {
    "planner_type": "<string>",
    "tools": ["<string>", ...],
    "max_steps": <int>,
    "stateless": <bool>
  },

  "optimization": {
    "runs": [<OptimizationRun>, ...]
  }
}
```

### Field Reference

#### `prompts`

| Field | Type | Description |
|-------|------|-------------|
| `role_definition` | `string` | The agent's role/persona. Optimized by MIPROv2. |
| `tool_instructions` | `ToolInstruction[]` | Auto-generated from registered tools. Describes available tools. |
| `worker_instructions` | `WorkerInstruction[]` | Instructions for delegating to workers (empty for single agents). |
| `format_instructions` | `string[]` | Output format requirements (JSON structure, etc.). Optimized by MIPROv2. |
| `examples` | `string[]` | Few-shot examples showing complete ReAct traces. Populated by optimization. |

#### `prompts.tool_instructions[]`

```json
{
  "name": "safe_calculator",
  "description": "Evaluates mathematical expressions safely.",
  "parameters": {
    "expression": {
      "type": "string",
      "description": "Mathematical expression to evaluate"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Tool name used in actions (e.g., `"safe_calculator"`) |
| `description` | `string` | What the tool does |
| `parameters` | `object` | Parameter schema (name -> type/description) |

#### `prompts.worker_instructions[]`

```json
{
  "name": "Calculator",
  "role_description": "Handles all mathematical calculations."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Worker name to delegate to |
| `role_description` | `string` | What this worker does |

#### `agent`

| Field | Type | Description |
|-------|------|-------------|
| `planner_type` | `string` | Planner class name (e.g., `"ReActPlanner"`, `"SimpleReActPlanner"`) |
| `tools` | `string[]` | List of tool class names (e.g., `["SafeCalculatorTool"]`) |
| `max_steps` | `int` | Maximum ReAct loop iterations before stopping |
| `stateless` | `bool` | Whether to clear memory between runs |

### Example: Math Agent

```json
{
  "version": "1.0",
  "type": "agent",

  "prompts": {
    "role_definition": "You are a precise math assistant. You MUST use the safe_calculator tool for ALL calculations. Never compute answers in your head.",
    "tool_instructions": [
      {
        "name": "safe_calculator",
        "description": "Evaluates mathematical expressions safely using Python's AST parser. Supports: +, -, *, /, **, %, parentheses.",
        "parameters": {
          "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate (e.g., '2 + 2', '0.25 * 80')"
          }
        }
      }
    ],
    "worker_instructions": [],
    "format_instructions": [
      "# RESPONSE FORMAT\nYou MUST respond with a JSON object containing 'thought' and 'action' keys.\nThe 'action' object must have 'tool_name' and 'tool_input' keys.",
      "# FINAL ANSWER FORMAT\nWhen providing the final answer, use tool_name 'final_answer' and set tool_input to ONLY the numeric result."
    ],
    "examples": []
  },

  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },

  "agent": {
    "planner_type": "ReActPlanner",
    "tools": ["SafeCalculatorTool"],
    "max_steps": 10,
    "stateless": false
  }
}
```

---

## MultiAgent Config (type: "multi_agent")

For optimizing a `HierarchicalAgentRunner` with a manager and worker agents.

### Schema

```json
{
  "version": "1.0",
  "type": "multi_agent",

  "manager": {
    "prompts": {
      "role_definition": "<string>",
      "tool_instructions": [],
      "worker_instructions": [<WorkerInstruction>, ...],
      "format_instructions": ["<string>", ...],
      "examples": ["<string>", ...]
    },
    "model": {
      "adapter": "<string>",
      "model_name": "<string>"
    },
    "agent": {
      "planner_type": "<string>",
      "tools": [],
      "max_steps": <int>
    }
  },

  "workers": {
    "<worker_name>": {
      "prompts": {
        "role_definition": "<string>",
        "tool_instructions": [<ToolInstruction>, ...],
        "worker_instructions": [],
        "format_instructions": ["<string>", ...],
        "examples": ["<string>", ...]
      },
      "model": {
        "adapter": "<string>",
        "model_name": "<string>"
      },
      "agent": {
        "planner_type": "<string>",
        "tools": ["<string>", ...],
        "max_steps": <int>
      }
    }
  },

  "model": {
    "adapter": "<string>",
    "model_name": "<string>",
    "adapter_kwargs": {}
  },

  "max_delegation_steps": <int>,

  "optimization": {
    "runs": [<OptimizationRun>, ...]
  }
}
```

### Field Reference

#### `manager`

The manager agent coordinates tasks and delegates to workers:

| Field | Type | Description |
|-------|------|-------------|
| `prompts` | `object` | Manager's prompt configuration |
| `prompts.worker_instructions` | `array` | Lists available workers and their capabilities |
| `prompts.tool_instructions` | `array` | Usually empty (manager uses workers instead of tools) |
| `model` | `object` | Optional per-agent model override |
| `agent` | `object` | Manager agent settings |

#### `workers`

A dictionary mapping worker names to their configurations:

| Field | Type | Description |
|-------|------|-------------|
| `<worker_name>` | `object` | Full agent config for this worker |
| `prompts.tool_instructions` | `array` | Worker's specialized tools |
| `prompts.examples` | `array` | Worker-specific examples (if `optimize_workers=True`) |

#### `max_delegation_steps`

| Field | Type | Description |
|-------|------|-------------|
| `max_delegation_steps` | `int` | Maximum total delegation steps across all workers |

### Example: Research Team

```json
{
  "version": "1.0",
  "type": "multi_agent",

  "manager": {
    "prompts": {
      "role_definition": "You are a research team manager. Coordinate research tasks by delegating to DataGatherer for information collection, then Summarizer for synthesis.",
      "tool_instructions": [],
      "worker_instructions": [
        {
          "name": "DataGatherer",
          "role_description": "Searches the web for relevant information. Input: search topic. Output: raw search results."
        },
        {
          "name": "Summarizer",
          "role_description": "Synthesizes raw data into clear summaries. Input: raw data. Output: concise summary."
        }
      ],
      "format_instructions": [
        "Delegate tasks using: {\"thought\": \"...\", \"action\": {\"tool_name\": \"WorkerName\", \"tool_input\": \"task\"}}",
        "Use 'final_answer' when complete."
      ],
      "examples": []
    },
    "agent": {
      "planner_type": "ManagerPlanner",
      "tools": [],
      "max_steps": 8
    }
  },

  "workers": {
    "DataGatherer": {
      "prompts": {
        "role_definition": "You are a research data gatherer. Search the web for accurate, up-to-date information.",
        "tool_instructions": [
          {
            "name": "web_searcher",
            "description": "Searches the web for information.",
            "parameters": {"query": {"type": "string", "description": "Search query"}}
          }
        ],
        "format_instructions": [],
        "examples": []
      },
      "agent": {
        "planner_type": "SimpleReActPlanner",
        "tools": ["WebSearcherTool"],
        "max_steps": 5
      }
    },
    "Summarizer": {
      "prompts": {
        "role_definition": "You are a research summarizer. Synthesize raw data into clear, concise summaries.",
        "tool_instructions": [],
        "format_instructions": [],
        "examples": []
      },
      "agent": {
        "planner_type": "SimpleReActPlanner",
        "tools": [],
        "max_steps": 3
      }
    }
  },

  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },

  "max_delegation_steps": 15
}
```

---

## Training Examples

Training examples are stored in JSON files and loaded with `load_training_examples()`.

### Schema

```json
[
  {
    "inputs": {
      "user_input": "<string>"
    },
    "expected_output": "<string>",
    "full_trace": "<string>"
  }
]
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | `object` | Yes | Input fields. Must contain `user_input`. |
| `inputs.user_input` | `string` | Yes | The user's query or request. |
| `expected_output` | `string` | Yes | The expected final answer/output. |
| `full_trace` | `string` | Depends | Complete execution trace. **Required** for `agent` and `multi_agent`. |

### When is `full_trace` Required?

| Config Type | `full_trace` Required? | Reason |
|-------------|------------------------|--------|
| `simple_llm` | No | No agent workflow to demonstrate |
| `agent` | Yes (recommended) | Shows model proper tool usage patterns |
| `multi_agent` | Yes (required) | Shows manager delegation workflow |

### SimpleLLM Training Data (no full_trace)

```json
[
  {
    "inputs": {"user_input": "I love this product!"},
    "expected_output": "SENTIMENT: positive"
  },
  {
    "inputs": {"user_input": "Terrible experience."},
    "expected_output": "SENTIMENT: negative"
  }
]
```

### Agent Training Data (with full_trace)

```json
[
  {
    "inputs": {"user_input": "What is 25 percent of 80?"},
    "expected_output": "20",
    "full_trace": "# --- Example: Percentage Calculation ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80. I will use the calculator with the expression 0.25 * 80.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"0.25 * 80\"\n    }\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0. This is the answer.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"20\"\n    }\n}"
  }
]
```

---

## The full_trace Format

The `full_trace` field contains a complete execution trace showing the model how to use tools correctly. This is the **key** to successful agent optimization.

### Purpose

- Teaches the model the exact ReAct workflow pattern
- Shows proper JSON formatting for thoughts and actions
- Demonstrates tool usage with real observations
- Provides the complete reasoning chain from question to answer

### Structure

A `full_trace` follows this pattern:

```
# --- Example: <Description> ---
"<Original user question>"

{
    "thought": "<Reasoning about what to do>",
    "action": {
        "tool_name": "<tool_to_use>",
        "tool_input": "<input_to_tool>"
    }
}

Observation: <Tool output/result>

{
    "thought": "<Reasoning about the observation>",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "<final_response>"
    }
}
```

### Components Explained

| Component | Description |
|-----------|-------------|
| `# --- Example: <Description> ---` | Header identifying the example type |
| `"<Original question>"` | The user's input, quoted |
| `{"thought": ..., "action": ...}` | ReAct step: reasoning + action |
| `"thought"` | Model's reasoning about what to do next |
| `"action.tool_name"` | Which tool to use |
| `"action.tool_input"` | Input to pass to the tool |
| `Observation:` | Tool's output (what the agent sees) |
| `"final_answer"` | Special tool name to end the loop |

### Single-Step Example

For simple tasks that require only one tool call:

```
# --- Example: Division ---
"Calculate 144 / 12"

{
    "thought": "I need to divide 144 by 12. I will use the calculator.",
    "action": {
        "tool_name": "safe_calculator",
        "tool_input": "144 / 12"
    }
}

Observation: The result of '144 / 12' is 12.0

{
    "thought": "The calculator shows 144 divided by 12 equals 12.",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "12"
    }
}
```

### Multi-Step Example

For tasks requiring multiple tool calls:

```
# --- Example: Complex Calculation ---
"What is 20% of 500, then add 50?"

{
    "thought": "I need to calculate 20% of 500 first. That's 0.20 * 500.",
    "action": {
        "tool_name": "safe_calculator",
        "tool_input": "0.20 * 500"
    }
}

Observation: The result of '0.20 * 500' is 100.0

{
    "thought": "20% of 500 is 100. Now I need to add 50 to get the final answer.",
    "action": {
        "tool_name": "safe_calculator",
        "tool_input": "100 + 50"
    }
}

Observation: The result of '100 + 50' is 150.0

{
    "thought": "The final result is 150.",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "150"
    }
}
```

### Multi-Agent full_trace

For multi-agent systems, the trace shows manager delegating to workers:

```
# --- Example: Research Coordination ---
"What are the latest developments in quantum computing?"

{
    "thought": "I need to research quantum computing developments. I will delegate to DataGatherer first.",
    "action": {
        "tool_name": "DataGatherer",
        "tool_input": "latest quantum computing developments 2024"
    }
}

Observation: DataGatherer returned: "Recent news includes IBM's 1000+ qubit processor, Google's error correction breakthrough..."

{
    "thought": "I have raw research data. Now I will delegate to Summarizer to synthesize this.",
    "action": {
        "tool_name": "Summarizer",
        "tool_input": "Recent news includes IBM's 1000+ qubit processor, Google's error correction breakthrough..."
    }
}

Observation: Summarizer returned: "Recent developments in quantum computing include advances in error correction and increased qubit counts."

{
    "thought": "I have a well-synthesized summary. I can now provide the final answer.",
    "action": {
        "tool_name": "final_answer",
        "tool_input": "Recent developments in quantum computing include advances in error correction and increased qubit counts."
    }
}
```

### Best Practices for full_trace

1. **Be explicit in thoughts**: Show the reasoning, not just "I'll use the tool"
2. **Match your tools**: Use the exact tool names from your config
3. **Realistic observations**: Write observations that match what the tool would return
4. **Clean final answers**: The `final_answer` tool_input should be just the answer, no extra text
5. **Diverse examples**: Include different types of questions and reasoning patterns
6. **Error handling**: Optionally include examples showing recovery from tool errors

### Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Missing quotes around question | JSON parsing issues | Always quote: `"What is 2+2?"` |
| Wrong tool name | Model learns wrong patterns | Match tool names exactly |
| Extra text in final_answer | Answer includes explanations | Keep it clean: just the result |
| Skipping thought steps | Model doesn't learn reasoning | Always include meaningful thoughts |
| Inconsistent JSON format | Model produces malformed output | Use consistent `{"thought": ..., "action": ...}` |

---

## Optimization Provenance

The `optimization` section tracks the history of optimization runs.

### Schema

```json
{
  "optimization": {
    "runs": [
      {
        "timestamp": "<ISO 8601 datetime>",
        "optimizer": "bootstrap" | "mipro",
        "metric": "<string>",
        "training_data_hash": "<string>",
        "examples_before": <int>,
        "examples_after": <int>,
        "role_definition_changed": <bool>,
        "format_instructions_changed": <bool>,
        "optimizer_config": {
          "max_bootstrapped_demos": <int>,
          "max_labeled_demos": <int>,
          "mipro_auto": "light" | "medium" | "heavy" | null,
          "optimize_manager": <bool>,
          "optimize_workers": <bool>,
          "workers_optimized": ["<string>", ...]
        }
      }
    ]
  }
}
```

### Field Reference

#### `OptimizationRun`

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `string` | ISO 8601 datetime when optimization completed |
| `optimizer` | `string` | Algorithm used: `"bootstrap"` or `"mipro"` |
| `metric` | `string` | Name of evaluation metric used |
| `training_data_hash` | `string` | SHA256 hash prefix of training data file (for reproducibility) |
| `examples_before` | `int` | Number of examples before this run |
| `examples_after` | `int` | Number of examples after this run |
| `role_definition_changed` | `bool` | Whether MIPROv2 modified the role definition |
| `format_instructions_changed` | `bool` | Whether format instructions were modified |
| `optimizer_config` | `object` | Parameters passed to the optimizer |

#### `optimizer_config`

| Field | Type | Description |
|-------|------|-------------|
| `max_bootstrapped_demos` | `int` | Max demos to generate via bootstrap |
| `max_labeled_demos` | `int` | Max labeled demos for MIPRO |
| `mipro_auto` | `string\|null` | MIPROv2 intensity level (null for bootstrap) |
| `optimize_manager` | `bool` | (multi_agent only) Whether manager was optimized |
| `optimize_workers` | `bool` | (multi_agent only) Whether workers were optimized |
| `workers_optimized` | `string[]` | (multi_agent only) List of optimized worker names |

### Derived Properties

The `OptimizationProvenance` class provides these computed properties:

| Property | Type | Description |
|----------|------|-------------|
| `optimized` | `bool` | `True` if any optimization runs exist |
| `optimizer` | `string` | Optimizer from most recent run |
| `metric` | `string` | Metric from most recent run |
| `created_at` | `string` | Timestamp of first run |
| `last_optimized_at` | `string` | Timestamp of most recent run |

---

## CLI Reference

The `fair-optimize` CLI provides commands for initialization, optimization, testing, and inspection.

### Global Usage

```bash
fair-optimize <command> [options]
```

### Commands

#### `init` - Initialize Config

Create a new config file from a template.

```bash
fair-optimize init [--type TYPE] [--output PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--type`, `-t` | `agent` | Config type: `simple_llm`, `agent`, `multi_agent` |
| `--output`, `-o` | `{type}_config.json` | Output file path |

**Example:**
```bash
fair-optimize init --type agent --output my_agent.json
```

#### `optimize` - Run Optimization

Optimize an agent's prompts using training data.

```bash
fair-optimize optimize -c CONFIG -t TRAINING [OPTIONS]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--config`, `-c` | Yes | - | Path to agent config JSON |
| `--training`, `-t` | Yes | - | Path to training examples JSON |
| `--output`, `-o` | No | `{config}_optimized.json` | Output path for optimized config |
| `--optimizer` | No | `bootstrap` | Optimizer: `bootstrap` or `mipro` |
| `--metric` | No | `contains_answer` | Metric: `exact_match`, `contains_answer`, `numeric_accuracy`, `fuzzy_match` |
| `--max-demos` | No | `4` | Max bootstrapped demos |
| `--model` | No | (from config) | Override model name |
| `--adapter` | No | (from config) | Override adapter: `HuggingFaceAdapter`, `OpenAIAdapter`, `OllamaAdapter` |
| `--mipro-lm` | No | - | DSPy LM for MIPROv2 (e.g., `ollama_chat/llama3:8b`) |
| `--mipro-auto` | No | `light` | MIPROv2 intensity: `light`, `medium`, `heavy` |

**Examples:**
```bash
# Bootstrap optimization
fair-optimize optimize -c agent.json -t examples.json --optimizer bootstrap

# MIPRO optimization
fair-optimize optimize -c agent.json -t examples.json \
    --optimizer mipro \
    --mipro-lm "ollama_chat/llama3:8b" \
    --mipro-auto medium

# With custom metric
fair-optimize optimize -c agent.json -t examples.json --metric numeric_accuracy
```

#### `test` - Interactive Testing

Test an agent interactively or with a single input.

```bash
fair-optimize test -c CONFIG [-i INPUT] [--model MODEL] [--adapter ADAPTER]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--config`, `-c` | Yes | - | Path to agent config JSON |
| `--input`, `-i` | No | (interactive) | Single input to test |
| `--model` | No | (from config) | Override model name |
| `--adapter` | No | (from config) | Override adapter |

**Examples:**
```bash
# Interactive mode
fair-optimize test -c agent_optimized.json

# Single input
fair-optimize test -c agent_optimized.json -i "What is 25% of 80?"
```

#### `info` - Show Config Info

Display information about a config file.

```bash
fair-optimize info -c CONFIG [--verbose]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--config`, `-c` | Yes | - | Path to config JSON |
| `--verbose`, `-v` | No | false | Show full config details |

**Example:**
```bash
fair-optimize info -c agent_optimized.json -v
```

#### `compare` - Compare Configs

Compare two config files side-by-side.

```bash
fair-optimize compare CONFIG1 CONFIG2
```

**Example:**
```bash
fair-optimize compare agent_initial.json agent_optimized.json
```

#### `examples` - Create Training Template

Generate a template for training examples.

```bash
fair-optimize examples [--output PATH] [--count N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `examples.json` | Output file path |
| `--count`, `-n` | `5` | Number of example templates |

**Example:**
```bash
fair-optimize examples -o training_data.json -n 10
```

---

## Metrics Reference

Metrics evaluate whether agent outputs meet expectations. All metrics have the signature:

```python
metric(example, prediction, trace=None) -> bool | float
```

### Built-in Metrics

| Metric | Return Type | Description |
|--------|-------------|-------------|
| `exact_match` | `bool` | Exact string equality (strips whitespace) |
| `contains_answer` | `bool` | Expected output is substring of actual |
| `numeric_accuracy` | `bool` | Numeric comparison (extracts numbers) |
| `fuzzy_match` | `float` | Character-level Jaccard similarity (0-1) |
| `json_format_compliance` | `bool` | Detects if malformed JSON leaked through |
| `sentiment_format_metric` | `float` | Graduated metric for sentiment format (0-1) |
| `research_quality_metric` | `float` | Evaluates research output quality (0-1) |

### Metric Factories

Create customized metrics:

#### `format_compliance(prefix: str)`

Checks if output starts with given prefix.

```python
from fair_prompt_optimizer import format_compliance

metric = format_compliance("ANSWER:")  # Returns True if output starts with "ANSWER:"
```

#### `keyword_match(keywords: list)`

Checks if all keywords are present.

```python
from fair_prompt_optimizer import keyword_match

metric = keyword_match(["price", "total", "$"])
```

#### `combined_metric(*metrics, weights=None)`

Combine multiple metrics with optional weights.

```python
from fair_prompt_optimizer import combined_metric, format_compliance, numeric_accuracy

metric = combined_metric(
    format_compliance("ANSWER:"),
    numeric_accuracy,
    weights=[0.3, 0.7]
)
```

#### `create_metric(...)`

Convenience factory for common patterns.

```python
from fair_prompt_optimizer import create_metric

metric = create_metric(
    check_format="RESULT:",
    check_numeric=True,
    tolerance=0.1
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `check_format` | `str` | Required prefix |
| `check_contains` | `str` | Required substring |
| `check_numeric` | `bool` | Whether to check numeric accuracy |
| `check_keywords` | `list` | Required keywords |
| `tolerance` | `float` | Numeric tolerance (default: 0.01) |

### Custom Metrics

Create your own metric function:

```python
def my_metric(example, prediction, trace=None) -> bool:
    expected = example.expected_output
    actual = prediction.response

    # Your evaluation logic
    return expected.lower() in actual.lower()
```

---

## Python API Reference

### Config Classes

#### `OptimizedConfig`

Main config wrapper with optimization provenance.

```python
from fair_prompt_optimizer import OptimizedConfig, load_optimized_config, save_optimized_config

# Load from file
config = load_optimized_config("agent.json")

# Access properties
config.type                    # "simple_llm" | "agent" | "multi_agent"
config.prompts                 # Dict of prompt components
config.role_definition         # Role definition string
config.examples                # List of examples
config.format_instructions     # List of format instructions
config.optimization            # OptimizationProvenance instance

# Modify
config.role_definition = "New role..."
config.examples = ["Example 1", "Example 2"]

# Save
save_optimized_config(config, "agent_optimized.json")
# or
config.save("agent_optimized.json")
```

#### `TrainingExample`

Single training example structure.

```python
from fair_prompt_optimizer import TrainingExample, load_training_examples, save_training_examples

# Load from file
examples = load_training_examples("examples.json")

# Create programmatically
example = TrainingExample(
    inputs={"user_input": "What is 2+2?"},
    expected_output="4",
    full_trace="# --- Example ---\n..."
)

# Save
save_training_examples([example], "examples.json")
```

### Optimizers

#### `SimpleLLMOptimizer`

For optimizing raw LLM + system prompt.

```python
from fair_prompt_optimizer import SimpleLLMOptimizer, load_training_examples
from fairlib import HuggingFaceAdapter

llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
system_prompt = "You are a classifier..."

optimizer = SimpleLLMOptimizer(llm, system_prompt)

examples = load_training_examples("examples.json")

result = optimizer.compile(
    training_examples=examples,
    metric=format_compliance("SENTIMENT:"),
    optimizer="bootstrap",           # or "mipro"
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    mipro_auto="light",              # "light" | "medium" | "heavy"
    training_data_path="examples.json",
    dspy_lm=dspy_lm,                 # Required for mipro
)

result.save("classifier_optimized.json")

# Test
output = optimizer.test("I love this!")
```

#### `AgentOptimizer`

For optimizing fairlib SimpleAgent with tools.

```python
from fair_prompt_optimizer import AgentOptimizer, load_training_examples, numeric_accuracy
from fairlib.utils.config_manager import load_agent

llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
agent = load_agent("agent.json", llm)

optimizer = AgentOptimizer(agent)

examples = load_training_examples("examples.json")

result = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="mipro",
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    mipro_auto="medium",
    training_data_path="examples.json",
    dspy_lm=dspy.LM("ollama_chat/llama3:8b"),
)

result.save("agent_optimized.json")
```

#### `MultiAgentOptimizer`

For optimizing HierarchicalAgentRunner systems.

```python
from fair_prompt_optimizer import MultiAgentOptimizer, load_training_examples
from fairlib.utils.config_manager import load_multi_agent

llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
runner = load_multi_agent("research_team.json", llm)

optimizer = MultiAgentOptimizer(
    runner,
    optimize_manager=True,
    optimize_workers=True,
)

manager_examples = load_training_examples("manager_examples.json")
worker_examples = {
    "DataGatherer": load_training_examples("gatherer_examples.json"),
    "Summarizer": load_training_examples("summarizer_examples.json"),
}

result = optimizer.compile(
    training_examples=manager_examples,
    metric=research_quality_metric,
    worker_training_examples=worker_examples,
    worker_metrics={
        "DataGatherer": contains_answer,
        "Summarizer": research_quality_metric,
    },
    optimizer="mipro",
    max_bootstrapped_demos=2,
    mipro_auto="light",
    dspy_lm=dspy.LM("ollama_chat/llama3:8b"),
)

result.save("research_team_optimized.json")
```

### Compile Parameters

All optimizers accept these parameters in `compile()`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `training_examples` | `List[TrainingExample]` | Yes | Training data |
| `metric` | `Callable` | Yes | Evaluation function |
| `optimizer` | `str` | No | `"bootstrap"` or `"mipro"` (default: `"bootstrap"`) |
| `max_bootstrapped_demos` | `int` | No | Max demos to generate (default: 4) |
| `max_labeled_demos` | `int` | No | Max labeled demos for MIPRO (default: 4) |
| `mipro_auto` | `str` | No | MIPRO intensity: `"light"`, `"medium"`, `"heavy"` |
| `training_data_path` | `str` | No | Path to training data (for provenance) |
| `dspy_lm` | `dspy.LM` | MIPRO only | DSPy LM for instruction generation |

#### MultiAgentOptimizer Additional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `worker_training_examples` | `Dict[str, List[TrainingExample]]` | Per-worker training data |
| `worker_metrics` | `Dict[str, Callable]` | Per-worker custom metrics |
| `optimize_manager` | `bool` | Whether to optimize manager (default: True) |
| `optimize_workers` | `bool` | Whether to optimize workers (default: False) |

---

## Complete Examples

### Example 1: Sentiment Classifier (SimpleLLM)

```python
from fair_prompt_optimizer import (
    SimpleLLMOptimizer,
    load_training_examples,
    format_compliance,
)
from fairlib import HuggingFaceAdapter

# Setup
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
system_prompt = """You are a sentiment classifier.

# RESPONSE FORMAT (MANDATORY)
Respond with ONLY one of:
SENTIMENT: positive
SENTIMENT: negative
SENTIMENT: neutral

No explanations, just the label."""

# Training data (no full_trace needed)
training_data = [
    {"inputs": {"user_input": "I love this!"}, "expected_output": "SENTIMENT: positive"},
    {"inputs": {"user_input": "Terrible."}, "expected_output": "SENTIMENT: negative"},
    {"inputs": {"user_input": "It's okay."}, "expected_output": "SENTIMENT: neutral"},
]

# Optimize
optimizer = SimpleLLMOptimizer(llm, system_prompt)
result = optimizer.compile(
    training_examples=[TrainingExample.from_dict(d) for d in training_data],
    metric=format_compliance("SENTIMENT:"),
    optimizer="bootstrap",
    max_bootstrapped_demos=3,
)

result.save("classifier_optimized.json")
```

### Example 2: Math Agent (Agent)

```python
from fair_prompt_optimizer import (
    AgentOptimizer,
    load_training_examples,
    numeric_accuracy,
)
from fairlib.utils.config_manager import load_agent
import dspy

# Setup
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
agent = load_agent("math_agent.json", llm)

# Training data (with full_trace - see format above)
examples = load_training_examples("math_examples.json")

# DSPy LM for MIPRO
dspy_lm = dspy.LM("ollama_chat/llama3:8b", api_base="http://localhost:11434")

# Optimize
optimizer = AgentOptimizer(agent)
result = optimizer.compile(
    training_examples=examples,
    metric=numeric_accuracy,
    optimizer="mipro",
    max_bootstrapped_demos=4,
    mipro_auto="medium",
    dspy_lm=dspy_lm,
    training_data_path="math_examples.json",
)

result.save("math_agent_optimized.json")

# Verify
print(f"Examples added: {len(result.examples)}")
print(f"Role changed: {result.optimization.runs[-1].role_definition_changed}")
```

### Example 3: Research Team (MultiAgent)

```python
from fair_prompt_optimizer import (
    MultiAgentOptimizer,
    load_training_examples,
    research_quality_metric,
    contains_answer,
)
from fairlib.utils.config_manager import load_multi_agent
import dspy

# Setup
llm = HuggingFaceAdapter("dolphin3-qwen25-3b")
runner = load_multi_agent("research_team.json", llm)

# Training data
manager_examples = load_training_examples("manager_examples.json")
worker_examples = {
    "DataGatherer": load_training_examples("gatherer_examples.json"),
    "Summarizer": load_training_examples("summarizer_examples.json"),
}

# DSPy LM
dspy_lm = dspy.LM("ollama_chat/llama3:8b", api_base="http://localhost:11434")

# Optimize
optimizer = MultiAgentOptimizer(runner, optimize_manager=True, optimize_workers=True)
result = optimizer.compile(
    training_examples=manager_examples,
    metric=research_quality_metric,
    worker_training_examples=worker_examples,
    worker_metrics={
        "DataGatherer": contains_answer,
        "Summarizer": research_quality_metric,
    },
    optimizer="mipro",
    max_bootstrapped_demos=2,
    mipro_auto="light",
    dspy_lm=dspy_lm,
)

result.save("research_team_optimized.json")

# Check results
last_run = result.optimization.runs[-1]
print(f"Manager examples: {last_run.examples_before} -> {last_run.examples_after}")
print(f"Workers optimized: {last_run.optimizer_config.get('workers_optimized', [])}")
```

---

## Quick Reference

### Config Type -> Optimizer -> Training Data

| Config Type | Optimizer Class | Training Data Format |
|-------------|-----------------|---------------------|
| `simple_llm` | `SimpleLLMOptimizer` | `inputs` + `expected_output` |
| `agent` | `AgentOptimizer` | `inputs` + `expected_output` + `full_trace` |
| `multi_agent` | `MultiAgentOptimizer` | `inputs` + `expected_output` + `full_trace` |

### Optimizer -> What Gets Optimized

| Optimizer | Optimizes Examples | Optimizes Instructions |
|-----------|-------------------|------------------------|
| `bootstrap` | Yes | No |
| `mipro` | Yes | Yes |

### MIPROv2 Intensity Levels

| Mode | Trials | Speed | Best For |
|------|--------|-------|----------|
| `light` | ~10 | Fast | Quick iteration, testing |
| `medium` | ~25 | Moderate | Balanced optimization |
| `heavy` | ~50+ | Slow | Production, final tuning |

### CLI Quick Reference

```bash
# Initialize
fair-optimize init --type agent -o my_agent.json

# Optimize (bootstrap)
fair-optimize optimize -c my_agent.json -t examples.json

# Optimize (MIPRO)
fair-optimize optimize -c my_agent.json -t examples.json \
    --optimizer mipro --mipro-lm "ollama_chat/llama3:8b" --mipro-auto medium

# Test
fair-optimize test -c my_agent_optimized.json

# Info
fair-optimize info -c my_agent_optimized.json -v

# Compare
fair-optimize compare my_agent.json my_agent_optimized.json

# Generate example template
fair-optimize examples -o examples.json -n 10
```
