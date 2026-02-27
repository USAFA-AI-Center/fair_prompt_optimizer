# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fair_prompt_optimizer is a DSPy-powered prompt optimization utility for FAIR-LLM agents. It optimizes prompts by running actual agents on training examples and using DSPy's optimization algorithms (BootstrapFewShot, MIPROv2) to improve performance.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run all tests (143 tests)
pytest tests/ -v

# Skip slow tests (require real LLM)
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=fair_prompt_optimizer

# Lint
ruff check fair_prompt_optimizer/ tests/

# Format
black fair_prompt_optimizer/ tests/

# CLI — optimize an agent
fair-optimize optimize -c CONFIG.json -t TRAINING.json --optimizer bootstrap
fair-optimize optimize -c CONFIG.json -t TRAINING.json --optimizer mipro --mipro-lm 'ollama_chat/llama3:8b' --mipro-auto medium

# CLI — other commands
fair-optimize init --type agent --output my_config.json
fair-optimize test -c CONFIG.json -i "test input"
fair-optimize info -c CONFIG.json
fair-optimize compare config_v1.json config_v2.json
fair-optimize validate -c CONFIG.json -t TRAINING.json
```

## Architecture

### Three Optimization Levels — `fair_prompt_optimizer/optimizers/`

| Level | Class | File | Use Case | Training Data |
|-------|-------|------|----------|---------------|
| 1 | `SimpleLLMOptimizer` | `simple_llm.py` | Classification, formatting | `inputs` + `expected_output` |
| 2 | `AgentOptimizer` | `agent.py` | Tool-using agents, ReAct | + `full_trace` (required) |
| 3 | `MultiAgentOptimizer` | `multi_agent.py` | Manager + worker hierarchy | + `full_trace` (required) |

Shared utilities and section markers in `optimizers/base.py`.

### Optimization Algorithms

- **BootstrapFewShot** — Optimizes examples only. Fast (~1 min), best for small datasets (10-50 examples).
- **MIPROv2** — Optimizes instructions + examples. Requires DSPy-compatible LM. Modes: light/medium/heavy.

### Config & Provenance — `config.py`
- `OptimizedConfig` wraps fairlib's config format and adds `optimization` provenance field
- Provenance tracks every optimization run: algorithm, metrics, parameters, file hashes
- fairlib ignores the `optimization` field (backward compatible)
- `load_optimized_config()` / `save_optimized_config()` handle serialization

### Metrics — `metrics.py`
Built-in metrics: `exact_match`, `contains_answer`, `numeric_accuracy`, `fuzzy_match`, `json_format_compliance`, `format_compliance_score`, `sentiment_format_metric`, `research_quality_metric`. Factory: `create_metric()`, `combined_metric()`.

### CLI — `cli.py`
Entry point: `fair-optimize` (registered via setuptools). Commands: init, optimize, test, info, compare, validate.

### Integration with fairlib
Takes fairlib agents as input, extracts configs via `fairlib.utils.config_manager`, applies DSPy optimization, saves in fairlib-compatible JSON format.

## Coding Rules

- Lint with `ruff`, format with `black` — CI enforces both
- Mark tests requiring real LLM calls with `@pytest.mark.slow`
- Mark tests requiring fairlib with `@pytest.mark.requires_fairlib`
- New metrics must return `bool` (simple) or `float` 0.0-1.0 (graduated) — register in `metrics.py` and export from `__init__.py`
- New optimizers must inherit from the patterns in `optimizers/base.py`
- Config changes must preserve backward compatibility — never remove fields from `OptimizedConfig`
