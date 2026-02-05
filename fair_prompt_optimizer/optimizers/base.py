# fair_prompt_optimizer/optimizers/base.py
"""
Shared utilities and constants for DSPy-compatible modules and optimizers.

This module contains:
- Section markers for structured prompt optimization
- Prompt parsing and combination utilities
- Async execution helpers
- Memory management utilities
"""

import asyncio
import gc
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..config import OptimizedPrompts

logger = logging.getLogger(__name__)

# Section markers for structured prompt optimization
ROLE_START = "<ROLE_DEFINITION>"
ROLE_END = "</ROLE_DEFINITION>"
FORMAT_START = "<FORMAT_INSTRUCTIONS>"
FORMAT_END = "</FORMAT_INSTRUCTIONS>"
FORMAT_ITEM_START = "<FORMAT_ITEM>"
FORMAT_ITEM_END = "</FORMAT_ITEM>"


def combine_prompt_components(
    role_definition: str,
    format_instructions: Optional[List[str]] = None,
) -> str:
    """
    Combine prompt components into a single structured string for MIPRO optimization.

    Uses XML-like markers so we can parse the optimized result back into components.
    """
    parts = []

    parts.append(f"{ROLE_START}")
    parts.append(role_definition.strip())
    parts.append(f"{ROLE_END}")

    if format_instructions:
        parts.append("")
        parts.append(f"{FORMAT_START}")
        for instruction in format_instructions:
            if isinstance(instruction, dict):
                text = instruction.get("text", instruction.get("content", str(instruction)))
            else:
                text = str(instruction)
            parts.append(f"{FORMAT_ITEM_START}")
            parts.append(text.strip())
            parts.append(f"{FORMAT_ITEM_END}")
        parts.append(f"{FORMAT_END}")

    return "\n".join(parts)


def parse_optimized_prompt(
    optimized_text: str,
    original_role: str,
    original_format_instructions: Optional[List[str]] = None,
) -> OptimizedPrompts:
    """
    Parse MIPRO's optimized instruction text back into prompt components.

    If parsing fails for any section, falls back to original values.
    """
    result = OptimizedPrompts()

    # Flexible role parsing - handle LLM typos in closing tag
    role_pattern = r"<ROLE_DEFINITION>\s*(.+?)\s*</ROLE_\w*>"
    role_match = re.search(role_pattern, optimized_text, re.DOTALL | re.IGNORECASE)

    if role_match:
        result.role_definition = role_match.group(1).strip()
        result.role_definition_changed = result.role_definition != original_role.strip()
    else:
        # Fallback - keep original
        result.role_definition = original_role
        result.role_definition_changed = False
        logger.warning("Could not parse ROLE_DEFINITION from MIPRO output")

    # Parse format instructions
    format_match = re.search(
        f"{re.escape(FORMAT_START)}\\s*(.+?)\\s*{re.escape(FORMAT_END)}", optimized_text, re.DOTALL
    )
    if format_match:
        format_content = format_match.group(1)
        items = re.findall(
            f"{re.escape(FORMAT_ITEM_START)}\\s*(.+?)\\s*{re.escape(FORMAT_ITEM_END)}",
            format_content,
            re.DOTALL,
        )
        if items:
            result.format_instructions = [item.strip() for item in items]
            if original_format_instructions:
                original_texts = [
                    (
                        fi.get("text", fi.get("content", str(fi)))
                        if isinstance(fi, dict)
                        else str(fi)
                    ).strip()
                    for fi in original_format_instructions
                ]
                result.format_instructions_changed = result.format_instructions != original_texts
            else:
                result.format_instructions_changed = True
    else:
        result.format_instructions = original_format_instructions
        result.format_instructions_changed = False
        logger.debug("No FORMAT_INSTRUCTIONS markers found, keeping original")

    return result


def run_async(coro):
    """
    Run an async coroutine from sync code.

    Handles the case where we might already be in an async context.
    """
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(coro)


def clear_cuda_memory():
    """Clear CUDA memory cache if torch is available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
