# JSON Contracts for fair_prompt_optimizer

Each optimizer produces a JSON config file that can be loaded back for inference or further optimization.

---

## 1. Agent Config (`type: "agent"`)

Used by `AgentOptimizer` for fairlib `SimpleAgent` instances.

```json
{
  "version": "1.0",
  "type": "agent",
  "prompts": {
    "role_definition": "You are a precise math assistant. You MUST use the safe_calculator tool for ALL calculations.",
    "format_instructions": [
      "Your response MUST be a valid JSON object with 'thought' and 'action' keys..."
    ],
    "tool_instructions": [],
    "worker_instructions": [],
    "examples": [
      "# --- Example: Percentage Calculation ---\n\"What is 25 percent of 80?\"\n\n{\n    \"thought\": \"I need to calculate 25 percent of 80.\",\n    \"action\": {\"tool_name\": \"safe_calculator\", \"tool_input\": \"0.25 * 80\"}\n}\n\nObservation: The result of '0.25 * 80' is 20.0\n\n{\n    \"thought\": \"The calculator returned 20.0.\",\n    \"action\": {\"tool_name\": \"final_answer\", \"tool_input\": \"25 percent of 80 is 20.\"}\n}"
    ]
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "cognitivecomputations/Dolphin3.0-Qwen2.5-3b",
    "adapter_kwargs": {}
  },
  "agent": {
    "agent_type": "SimpleAgent",
    "planner_type": "ReActPlanner",
    "tools": ["SafeCalculatorTool"],
    "max_steps": 10,
    "stateless": false
  },
  "optimization": {
    "optimized": true,
    "optimizer": "bootstrap",
    "metric": "numeric_accuracy",
    "training_data": {
      "path": "/path/to/training_data.json",
      "hash": "sha256:abc123...",
      "num_examples": 12
    },
    "runs": [
      {
        "timestamp": "2026-01-13T19:14:12.013448",
        "optimizer": "bootstrap",
        "metric": "numeric_accuracy",
        "training_data_hash": "sha256:abc123...",
        "examples_before": 0,
        "examples_after": 4,
        "role_definition_changed": false,
        "format_instructions_changed": false,
        "optimizer_config": {
          "max_bootstrapped_demos": 4,
          "max_labeled_demos": 4
        }
      }
    ],
    "created_at": "2026-01-13T19:14:12.013448",
    "last_optimized_at": "2026-01-13T19:14:12.013448"
  },
  "metadata": {
    "exported_at": "2026-01-13T19:14:12.013448",
    "source": "fair_prompt_optimizer"
  }
}
```

### Key Points for Agent Config:
- **`prompts.examples` MUST be full_trace** showing complete tool workflow
- Examples show: question → thought/action JSON → Observation → final_answer
- MIPRO may update `role_definition` but examples are ALWAYS full_trace

---

## 2. SimpleLLM Config (`type: "simple_llm"`)

Used by `SimpleLLMOptimizer` for plain LLM + system prompt (no agent/tools).

```json
{
  "version": "1.0",
  "type": "simple_llm",
  "prompts": {
    "system_prompt": "You are a sentiment classifier.\n\nRespond with ONLY one of:\nSENTIMENT: positive\nSENTIMENT: negative\nSENTIMENT: neutral",
    "examples": [
      "Example 1:\nInput: I love this product!\nOutput: SENTIMENT: positive",
      "Example 2:\nInput: This is terrible.\nOutput: SENTIMENT: negative",
      "Example 3:\nInput: It's okay.\nOutput: SENTIMENT: neutral"
    ]
  },
  "model": {
    "adapter": "HuggingFaceAdapter",
    "model_name": "cognitivecomputations/Dolphin3.0-Qwen2.5-3b",
    "adapter_kwargs": {}
  },
  "optimization": {
    "optimized": true,
    "optimizer": "bootstrap",
    "metric": "format_compliance",
    "training_data": {
      "path": "/path/to/training_data.json",
      "hash": "sha256:def456...",
      "num_examples": 5
    },
    "runs": [
      {
        "timestamp": "2026-01-13T20:00:00.000000",
        "optimizer": "bootstrap",
        "metric": "format_compliance",
        "examples_before": 0,
        "examples_after": 3,
        "role_definition_changed": false
      }
    ]
  },
  "metadata": {
    "exported_at": "2026-01-13T20:00:00.000000",
    "source": "fair_prompt_optimizer"
  }
}
```

### Key Points for SimpleLLM Config:
- **No full_trace needed** - simple input/output pairs work fine
- DSPy BootstrapFewShot/MIPROv2 handle example selection
- No `agent` section (no tools, no planner)
- Use for: classification, format compliance, simple generation

---

## 3. MultiAgent Config (`type: "multi_agent"`)

Used by `MultiAgentOptimizer` for fairlib `HierarchicalAgentRunner`.

```json
{
  "version": "1.0",
  "type": "multi_agent",
  "manager": {
    "prompts": {
      "role_definition": "You are a manager that delegates tasks to specialized workers.",
      "format_instructions": ["..."],
      "worker_instructions": [
        {"name": "Calculator", "role_description": "Handles math calculations."}
      ],
      "examples": [
        "# --- Example: Multi-Agent Math ---\n\"What is 50% of 80?\"\n\nManager thought: I need to delegate this to Calculator.\n..."
      ]
    },
    "agent": {
      "agent_type": "SimpleAgent",
      "planner_type": "ReActPlanner",
      "max_steps": 5
    }
  },
  "workers": {
    "Calculator": {
      "prompts": {
        "role_definition": "You are a calculator specialist.",
        "examples": ["<worker_full_trace>"]
      },
      "agent": {
        "agent_type": "SimpleAgent",
        "planner_type": "ReActPlanner",
        "tools": ["SafeCalculatorTool"],
        "max_steps": 5
      }
    }
  },
  "optimization": {
    "optimized": true,
    "optimizer": "bootstrap",
    "metric": "contains_answer",
    "runs": [...]
  },
  "metadata": {
    "exported_at": "2026-01-13T20:30:00.000000",
    "source": "fair_prompt_optimizer"
  }
}
```

### Key Points for MultiAgent Config:
- **`examples` MUST be full_trace** for both manager and workers
- Each worker can have its own full_trace showing its workflow
- Manager trace shows delegation decisions
- MIPRO may update manager's `role_definition`

---

## Summary Table

| Type | full_trace Required? | DSPy Used For |
|------|---------------------|---------------|
| `agent` | **YES** | MIPRO: role_definition only |
| `simple_llm` | No | Bootstrap + MIPRO: examples + system_prompt |
| `multi_agent` | **YES** | MIPRO: manager role_definition only |

---

## Loading Configs

```python
from fair_prompt_optimizer import load_optimized_config

# Load any config type
config = load_optimized_config("agent_optimized.json")

# Check type
if config.config["type"] == "agent":
    # Use with fairlib load_agent
    from fairlib.utils.config_manager import load_agent
    agent = load_agent("agent_optimized.json", llm)
    
elif config.config["type"] == "simple_llm":
    # Use system_prompt + examples directly
    system_prompt = config.prompts["system_prompt"]
    examples = config.prompts["examples"]
```