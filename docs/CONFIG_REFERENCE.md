# Configuration Reference

Complete documentation for all configuration structures in `fair_prompt_optimizer`.

## Table of Contents

1. [Overview](#overview)
2. [Config Types](#config-types)
3. [SimpleLLM Config](#simplellm-config-type-simple_llm)
4. [Agent Config](#agent-config-type-agent)
5. [MultiAgent Config](#multiagent-config-type-multi_agent)
6. [Training Examples](#training-examples)
7. [Optimization Provenance](#optimization-provenance)
8. [Complete Examples](#complete-examples)

---

## Overview

All configs share a common structure with type-specific sections:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONFIG STRUCTURE                           │
├─────────────────────────────────────────────────────────────────┤
│  version        │  Config format version (e.g., "1.0")         │
│  type           │  "simple_llm" | "agent" | "multi_agent"      │
│  prompts        │  Prompt components (role, examples, etc.)    │
│  model          │  LLM adapter configuration                   │
│  agent          │  Agent-specific settings (if applicable)     │
│  optimization   │  Provenance tracking (added after optimize)  │
└─────────────────────────────────────────────────────────────────┘
```

### Config Lifecycle

```
┌──────────────┐    optimize()    ┌───────────────────┐
│ Initial      │ ──────────────►  │ Optimized         │
│ Config       │                  │ Config            │
├──────────────┤                  ├───────────────────┤
│ • prompts    │                  │ • prompts         │
│   └ examples │                  │   └ examples ←─── │ (populated few-shot examples)
│     = []     │                  │     = [...]       │
│              │                  │ • optimization    │
│              │                  │   └ runs: [...]   │
└──────────────┘                  └───────────────────┘
```

---

## Config Types

| Type | Optimizer Class | Use Case | Requires `full_trace` |
|------|-----------------|----------|----------------------|
| `simple_llm` | `SimpleLLMOptimizer` | Classification, format compliance, simple generation | No |
| `agent` | `AgentOptimizer` | Tool-using agents, ReAct workflows | Yes (recommended) |
| `multi_agent` | `MultiAgentOptimizer` | Manager + worker hierarchies | Yes (required) |

---

## SimpleLLM Config (type: "simple_llm")

For optimizing a raw LLM with a system prompt—no agent pipeline or tools.

### Schema

```json
{
  "version": "1.0",
  "type": "simple_llm",
  
  "prompts": {
    "system_prompt": "<string>",
    "examples": ["<string>", ...]
  },
  
  "model": {
    "adapter": "<string>",
    "model_name": "<string>",
    "adapter_kwargs": {}
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
| `system_prompt` | `string` | The system prompt/instructions for the LLM. This is what gets optimized by MIPROv2. |
| `examples` | `string[]` | Few-shot examples added by optimization. Each is a formatted input/output pair. |

#### `model`

| Field | Type | Description |
|-------|------|-------------|
| `adapter` | `string` | LLM adapter class name. One of: `HuggingFaceAdapter`, `OpenAIAdapter`, `OllamaAdapter` |
| `model_name` | `string` | Model identifier (e.g., `"dolphin3-qwen25-3b"`, `"gpt-4o"`, `"llama3:8b"`) |
| `adapter_kwargs` | `object` | Additional kwargs passed to adapter constructor |

### Example: Sentiment Classifier

```json
{
  "version": "1.0",
  "type": "simple_llm",
  
  "prompts": {
    "system_prompt": "You are a sentiment classifier.\n\n# RESPONSE FORMAT (MANDATORY)\nRespond with ONLY one of:\nSENTIMENT: positive\nSENTIMENT: negative\nSENTIMENT: neutral\n\nNo explanations, just the label.",
    "examples": []
  },
  
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  }
}
```

### After Optimization

```json
{
  "version": "1.0",
  "type": "simple_llm",
  
  "prompts": {
    "system_prompt": "You are a sentiment classifier.\n\n# RESPONSE FORMAT (MANDATORY)\nRespond with ONLY one of:\nSENTIMENT: positive\nSENTIMENT: negative\nSENTIMENT: neutral\n\nNo explanations, just the label.",
    "examples": [
      {
        "user_input": "I absolutely love this product!",
        "expected_output": "SENTIMENT: positive"
      },
      {
        "user_input": "This is terrible. Complete waste of money.",
        "expected_output": "SENTIMENT: negative"
      },
      {
        "user_input": "It's okay, nothing special.",
        "expected_output": "SENTIMENT: neutral"
      }
    ]
  },
  
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },
  
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-14T10:30:00.123456",
        "optimizer": "bootstrap",
        "metric": "format_compliance_SENTIMENT:",
        "training_data_hash": "sha256:a1b2c3d4e5f6",
        "examples_before": 0,
        "examples_after": 3,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 3,
          "max_labeled_demos": 4,
          "mipro_auto": null
        }
      }
    ]
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
    "max_steps": <int>
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
| `tool_instructions` | `ToolInstruction[]` | Auto-generated from registered tools. |
| `worker_instructions` | `WorkerInstruction[]` | Instructions for delegating to workers (if any). |
| `format_instructions` | `string[]` | Output format requirements (JSON structure, etc.). |
| `examples` | `string[]` | Few-shot examples showing complete ReAct traces. |

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

#### `prompts.worker_instructions[]`

```json
{
  "name": "Calculator",
  "role_description": "Handles all mathematical calculations."
}
```

#### `agent`

| Field | Type | Description |
|-------|------|-------------|
| `planner_type` | `string` | Planner class name (e.g., `"ReActPlanner"`) |
| `tools` | `string[]` | List of tool class names (e.g., `["SafeCalculatorTool"]`) |
| `max_steps` | `int` | Maximum ReAct loop iterations |

### Example: Math Agent (Initial)

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
      "# RESPONSE FORMAT\nYou MUST respond with a JSON object containing 'thought' and 'action' keys.\nThe 'action' object must have 'tool_name' and 'tool_input' keys.\n\nExample:\n{\"thought\": \"I need to calculate X\", \"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"2 + 2\"}}",
      "# FINAL ANSWER FORMAT\nWhen providing the final answer, use tool_name 'final_answer' and set tool_input to ONLY the numeric result (e.g., '42', not 'The answer is 42')."
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
    "max_steps": 10
  }
}
```

### After Optimization (with full_trace examples)

```json
{
  "version": "1.0",
  "type": "agent",
  
  "prompts": {
    "role_definition": "You are a precise math assistant. You MUST use the safe_calculator tool for ALL calculations. Never compute answers in your head.",
    "tool_instructions": [
      {
        "name": "safe_calculator",
        "description": "Evaluates mathematical expressions safely using Python's AST parser.",
        "parameters": {
          "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate"
          }
        }
      }
    ],
    "worker_instructions": [],
    "format_instructions": [
      "# RESPONSE FORMAT\nYou MUST respond with a JSON object containing 'thought' and 'action' keys..."
    ],
    "examples": [
      "# --- Example: Percentage Calculation ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80. I will use the calculator with the expression 0.25 * 80.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"0.25 * 80\"\n    }\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0. This is the answer.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"20\"\n    }\n}",
      "# --- Example: Division ---\n\"Calculate 144 / 12\"\n\n{\n    \"thought\": \"I need to divide 144 by 12. I will use the calculator.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"144 / 12\"\n    }\n}\n\nObservation: The result of '144 / 12' is 12.0\n\n{\n    \"thought\": \"The calculator shows 144 divided by 12 equals 12.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"12\"\n    }\n}"
    ]
  },
  
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },
  
  "agent": {
    "planner_type": "ReActPlanner",
    "tools": ["SafeCalculatorTool"],
    "max_steps": 10
  },
  
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-14T10:30:00.123456",
        "optimizer": "mipro",
        "metric": "numeric_accuracy",
        "training_data_hash": "sha256:b2c3d4e5f6a7",
        "examples_before": 0,
        "examples_after": 2,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 4,
          "max_labeled_demos": 4,
          "mipro_auto": "medium"
        }
      }
    ]
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
  
  "optimization": {
    "runs": [<OptimizationRun>, ...]
  }
}
```

### Field Reference

#### `manager`

The manager agent coordinates tasks and delegates to workers. It typically has:
- `worker_instructions` listing available workers
- No `tools` (uses workers instead)
- `format_instructions` for delegation format

#### `workers`

A dictionary mapping worker names to their configurations. Each worker:
- Has specialized `tools` for its domain
- Has a focused `role_definition`
- May have its own `examples` (if `optimize_workers=True`)

### Example: Calculator Multi-Agent (Initial)

```json
{
  "version": "1.0",
  "type": "multi_agent",
  
  "manager": {
    "prompts": {
      "role_definition": "You are a manager that delegates math problems to the Calculator worker.",
      "tool_instructions": [],
      "worker_instructions": [
        {
          "name": "Calculator",
          "role_description": "Handles all mathematical calculations. Send math expressions to this worker."
        }
      ],
      "format_instructions": [
        "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"Calculator\", \"tool_input\": \"the math task\"}}"
      ],
      "examples": []
    },
    "agent": {
      "planner_type": "ReActPlanner",
      "tools": [],
      "max_steps": 5
    }
  },
  
  "workers": {
    "Calculator": {
      "prompts": {
        "role_definition": "You are a calculator specialist. Use safe_calculator for all math.",
        "tool_instructions": [
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
        ],
        "worker_instructions": [],
        "format_instructions": [
          "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"...\", \"tool_input\": \"...\"}}"
        ],
        "examples": []
      },
      "agent": {
        "planner_type": "ReActPlanner",
        "tools": ["SafeCalculatorTool"],
        "max_steps": 5
      }
    }
  },
  
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  }
}
```

### After Optimization

```json
{
  "version": "1.0",
  "type": "multi_agent",
  
  "manager": {
    "prompts": {
      "role_definition": "You are a manager that delegates math problems to the Calculator worker.",
      "tool_instructions": [],
      "worker_instructions": [
        {
          "name": "Calculator",
          "role_description": "Handles all mathematical calculations."
        }
      ],
      "format_instructions": [
        "Respond with JSON: {\"thought\": \"...\", \"action\": {\"tool_name\": \"Calculator\", \"tool_input\": \"task\"}}"
      ],
      "examples": [
        "# --- Example: Percentage ---\n\"What is 50% of 80?\"\n\nManager:\n{\"thought\": \"This is a math problem. I'll delegate to Calculator.\", \"action\": {\"tool_name\": \"Calculator\", \"tool_input\": \"Calculate 50% of 80\"}}\n\nCalculator Response: 40\n\nManager:\n{\"thought\": \"Calculator returned 40.\", \"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"40\"}}"
      ]
    },
    "agent": {
      "planner_type": "ReActPlanner",
      "tools": [],
      "max_steps": 5
    }
  },
  
  "workers": {
    "Calculator": {
      "prompts": {
        "role_definition": "You are a calculator specialist. Use safe_calculator for all math.",
        "tool_instructions": [
          {
            "name": "safe_calculator",
            "description": "Evaluates mathematical expressions safely.",
            "parameters": {}
          }
        ],
        "worker_instructions": [],
        "format_instructions": [],
        "examples": []
      },
      "agent": {
        "planner_type": "ReActPlanner",
        "tools": ["SafeCalculatorTool"],
        "max_steps": 5
      }
    }
  },
  
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },
  
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-14T11:00:00.123456",
        "optimizer": "bootstrap",
        "metric": "contains_answer",
        "training_data_hash": "sha256:c3d4e5f6a7b8",
        "examples_before": 0,
        "examples_after": 1,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 2,
          "max_labeled_demos": 4,
          "mipro_auto": null,
          "optimize_manager": true,
          "optimize_workers": false
        }
      }
    ]
  }
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
    "full_trace": "<string>  (required if optimizing an Agent or Multi-agent system)"
  }
]
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | `object` | Yes | Input fields. Must contain `user_input`. |
| `inputs.user_input` | `string` | Yes | The user's query or request. |
| `expected_output` | `string` | Yes | The expected final answer/output. |
| `full_trace` | `string` | Depends | Complete execution trace showing thought/action/observation cycles. **Required** for `agent` and `multi_agent` optimization. |

### Example: SimpleLLM Training Data (no full_trace)

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

### Example: Agent Training Data (with full_trace)

```json
[
  {
    "inputs": {"user_input": "What is 25 percent of 80?"},
    "expected_output": "20",
    "full_trace": "# --- Example: Percentage Calculation ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80. I will use the calculator with the expression 0.25 * 80.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"0.25 * 80\"\n    }\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0. This is the answer.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"20\"\n    }\n}"
  }
]
```

### full_trace Format

TODO:: BUILD THIS OUT

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
          "optimize_manager": <bool>,      // multi_agent only
          "optimize_workers": <bool>       // multi_agent only
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
| `training_data_hash` | `string` | SHA256 hash prefix of training data file |
| `examples_before` | `int` | Number of examples before this run |
| `examples_after` | `int` | Number of examples after this run |
| `role_definition_changed` | `bool` | Whether MIPROv2 modified the role definition |
| `format_instructions_changed` | `bool` | Whether format instructions were modified |
| `optimizer_config` | `object` | Parameters passed to the optimizer |

### Derived Properties

The `OptimizationProvenance` class provides these computed properties:

| Property | Type | Description |
|----------|------|-------------|
| `optimized` | `bool` | `True` if any optimization runs exist |
| `optimizer` | `string` | Optimizer from most recent run |
| `metric` | `string` | Metric from most recent run |
| `created_at` | `string` | Timestamp of first run |
| `last_optimized_at` | `string` | Timestamp of most recent run |

### Example: Multiple Optimization Runs

```json
{
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-10T09:00:00.000000",
        "optimizer": "bootstrap",
        "metric": "numeric_accuracy",
        "training_data_hash": "sha256:a1b2c3d4",
        "examples_before": 0,
        "examples_after": 3,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 4,
          "max_labeled_demos": 4,
          "mipro_auto": null
        }
      },
      {
        "timestamp": "2025-01-14T10:30:00.000000",
        "optimizer": "mipro",
        "metric": "numeric_accuracy",
        "training_data_hash": "sha256:e5f6a7b8",
        "examples_before": 3,
        "examples_after": 5,
        "role_definition_changed": true,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 4,
          "max_labeled_demos": 4,
          "mipro_auto": "medium"
        }
      }
    ]
  }
}
```

---

## Complete Examples

### SimpleLLM: Sentiment Classifier (Full)

<details>
<summary>classifier_config.json</summary>

```json
{
  "version": "1.0",
  "type": "simple_llm",
  "prompts": {
    "system_prompt": "You are a sentiment classifier.\n\n# RESPONSE FORMAT (MANDATORY)\nRespond with ONLY one of:\nSENTIMENT: positive\nSENTIMENT: negative\nSENTIMENT: neutral\n\nNo explanations, just the label.",
    "examples": [
      {
        "user_input": "I absolutely love this product! Best purchase I've ever made.",
        "expected_output": "SENTIMENT: positive"
      },
      {
        "user_input": "This is terrible. Complete waste of money.",
        "expected_output": "SENTIMENT: negative"
      },
      {
        "user_input": "It's okay, nothing special really.",
        "expected_output": "SENTIMENT: neutral"
      }
    ]
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-14T10:30:00.123456",
        "optimizer": "bootstrap",
        "metric": "format_compliance_SENTIMENT:",
        "training_data_hash": "sha256:a1b2c3d4e5f6",
        "examples_before": 0,
        "examples_after": 3,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 3,
          "max_labeled_demos": 4,
          "mipro_auto": null
        }
      }
    ]
  }
}
```

</details>

### Agent: Math Calculator (Full)

<details>
<summary>math_agent_config.json</summary>

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
      "# RESPONSE FORMAT\nYou MUST respond with a JSON object containing 'thought' and 'action' keys.\nThe 'action' object must have 'tool_name' and 'tool_input' keys.\n\nExample:\n{\"thought\": \"I need to calculate X\", \"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"2 + 2\"}}",
      "# FINAL ANSWER FORMAT\nWhen providing the final answer, use tool_name 'final_answer' and set tool_input to ONLY the numeric result (e.g., '42', not 'The answer is 42')."
    ],
    "examples": [
      "# --- Example: Percentage Calculation ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80. I will use the calculator with the expression 0.25 * 80.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"0.25 * 80\"\n    }\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0. This is the answer.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"20\"\n    }\n}",
      "# --- Example: Division ---\n\"Calculate 144 / 12\"\n\n{\n    \"thought\": \"I need to divide 144 by 12. I will use the calculator.\",\n    \"action\": {\n        \"tool_name\": \"safe_calculator\",\n        \"tool_input\": \"144 / 12\"\n    }\n}\n\nObservation: The result of '144 / 12' is 12.0\n\n{\n    \"thought\": \"The calculator shows 144 divided by 12 equals 12.\",\n    \"action\": {\n        \"tool_name\": \"final_answer\",\n        \"tool_input\": \"12\"\n    }\n}"
    ]
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "dolphin3-qwen25-3b",
    "adapter_kwargs": {}
  },
  "agent": {
    "planner_type": "ReActPlanner",
    "tools": ["SafeCalculatorTool"],
    "max_steps": 10
  },
  "optimization": {
    "runs": [
      {
        "timestamp": "2025-01-14T10:30:00.123456",
        "optimizer": "mipro",
        "metric": "numeric_accuracy",
        "training_data_hash": "sha256:b2c3d4e5f6a7",
        "examples_before": 0,
        "examples_after": 2,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 4,
          "max_labeled_demos": 4,
          "mipro_auto": "medium"
        }
      }
    ]
  }
}
```

</details>

---

## Quick Reference

### Config Type → Optimizer → Training Data

| Config Type | Optimizer Class | Training Data Format |
|-------------|-----------------|---------------------|
| `simple_llm` | `SimpleLLMOptimizer` | `inputs` + `expected_output` |
| `agent` | `AgentOptimizer` | `inputs` + `expected_output` + `full_trace` |
| `multi_agent` | `MultiAgentOptimizer` | `inputs` + `expected_output` + `full_trace` |

### Optimizer → What Gets Optimized

| Optimizer | Optimizes Examples | Optimizes Instructions |
|-----------|-------------------|------------------------|
| `bootstrap` |  Yes |  No |
| `mipro` |  Yes |  Yes |

### MIPROv2 Intensity Levels

| Mode | Trials | Speed | Best For |
|------|--------|-------|----------|
| `light` | ~10 | Fast | Quick iteration, testing |
| `medium` | ~25 | Moderate | Balanced optimization |
| `heavy` | ~50+ | Slow | Production, final tuning |