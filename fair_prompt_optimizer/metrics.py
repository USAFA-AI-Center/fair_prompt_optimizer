# fair_prompt_optimizer/metrics.py
"""
Built-in metrics for evaluating optimization quality.

Metrics are functions with signature:
    metric(example, prediction, trace=None) -> bool | float

Where:
- example: The training example (has .expected_output, .inputs, etc.)
- prediction: The model's prediction (has .response or output field)
- trace: Optional trace info from DSPy

Returns True/False for binary metrics, or float 0-1 for continuous.
"""

import re
from typing import Callable, Optional


def _get_attr(obj, attr: str, default: str = '') -> str:
    """Get attribute from object or dict."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def json_format_compliance(example, prediction, trace=None) -> bool:
    """
    Detects if malformed JSON leaked through to the final answer.
    
    Returns True if the output appears to be a clean final answer.
    Returns False if the output contains JSON fragments (indicating parse failure).
    
    TODO: This doesn't catch Case 2 where the model ignores format entirely
    and outputs natural language like "The answer is 20." That case requires
    instrumenting FinalAnswer.from_tool in fair_llm.
    """
    response = str(_get_attr(prediction, 'response')).strip()
    
    # Indicators that malformed JSON leaked through
    json_fragments = [
        '{"thought"',
        '{"action"',
        '"tool_name"',
        '"tool_input"',
        '{"thought":',
        '{"action":',
    ]
    
    # Check if response contains JSON structure fragments
    for fragment in json_fragments:
        if fragment in response:
            return False
    
    if response.startswith('{') and not response.endswith('}'):
        return False  # Truncated JSON
    
    if response.startswith('{') and 'thought' in response.lower():
        return False  # Looks like a ReAct response that leaked
    
    return True


def numeric_accuracy_with_format(example, prediction, trace=None, tolerance: float = 0.00) -> float:
    """
    Combined metric: answer must be correct AND properly formatted.
    
    Returns:
        1.0 - Correct answer via clean format
        0.5 - Correct answer but malformed JSON leaked (format failure)
        0.0 - Wrong answer
    
    TODO: Add Case 2 detection (natural language bypass) once 
    FinalAnswer.from_tool is implemented in fair_llm.
    """
    format_ok = json_format_compliance(example, prediction, trace)
    answer_ok = numeric_accuracy(example, prediction, trace, tolerance)
    
    if answer_ok and format_ok:
        return 1.0
    elif answer_ok and not format_ok:
        return 0.5  # Got lucky with fallback
    else:
        return 0.0


def exact_match(example, prediction, trace=None) -> bool:
    """
    Exact string match between expected and actual output.
    
    Strips whitespace and compares.
    """
    expected = str(_get_attr(example, 'expected_output')).strip()
    actual = str(_get_attr(prediction, 'response')).strip()
    return expected == actual


def contains_answer(example, prediction, trace=None) -> bool:
    """
    Check if expected output is contained in the prediction.
    
    Useful when the model adds extra explanation.
    """
    expected = str(_get_attr(example, 'expected_output')).strip().lower()
    actual = str(_get_attr(prediction, 'response')).strip().lower()
    return expected in actual


def numeric_accuracy(example, prediction, trace=None, tolerance: float = 0.00) -> bool:
    """
    Compare numeric values with tolerance.
    
    Extracts numbers from both strings and compares.
    """
    def extract_number(text: str) -> Optional[float]:
        # Find numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', str(text))
        return float(numbers[-1]) if numbers else None
    
    expected = str(_get_attr(example, 'expected_output'))
    actual = str(_get_attr(prediction, 'response'))
    
    expected_num = extract_number(expected)
    actual_num = extract_number(actual)
    
    if expected_num is None or actual_num is None:
        return False
    
    return abs(expected_num - actual_num) <= tolerance


def format_compliance(prefix: str) -> Callable:
    """
    Factory for format compliance metrics.
    
    Returns a metric that checks if output starts with the given prefix.
    
    Usage:
        metric = format_compliance("ANSWER:")
    """
    def metric(example, prediction, trace=None) -> bool:
        actual = str(_get_attr(prediction, 'response')).strip()
        return actual.upper().startswith(prefix.upper())
    
    metric.__name__ = f"format_compliance_{prefix}"
    return metric

def sentiment_format_metric(example, prediction, trace=None) -> float:
    """
    Graduated metric for sentiment FORMAT compliance only.
    
    We don't care if classification is correct - just the format.
    
    Returns:
        1.0 - Perfect format: "SENTIMENT: positive/negative/neutral"
        0.5 - Has "SENTIMENT" somewhere but not perfect format
        0.3 - Outputs a valid label (positive/negative/neutral) without prefix
        0.0 - Garbage output
    """
    actual = str(_get_attr(prediction, 'response')).strip().lower()
    
    valid_labels = ['positive', 'negative', 'neutral']
    
    for label in valid_labels:
        if actual == f"sentiment: {label}":
            return 1.0
    
    if actual.startswith("sentiment:"):
        return 0.8
    
    if "sentiment" in actual:
        return 0.5
    
    for label in valid_labels:
        if actual == label:
            return 0.3
        if label in actual and len(actual) < 50:
            return 0.2
    
    return 0.0


def fuzzy_match(example, prediction, trace=None, threshold: float = 0.8) -> float:
    """
    Fuzzy string matching using character-level similarity.
    
    Returns a score between 0 and 1.
    """
    expected = str(_get_attr(example, 'expected_output')).strip().lower()
    actual = str(_get_attr(prediction, 'response')).strip().lower()
    
    if not expected or not actual:
        return 0.0
    
    # Simple character-level Jaccard similarity
    set1 = set(expected)
    set2 = set(actual)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def keyword_match(keywords: list) -> Callable:
    """
    Factory for keyword matching metrics.
    
    Returns a metric that checks if all keywords are present.
    
    Usage:
        metric = keyword_match(["price", "total", "$"])
    """
    def metric(example, prediction, trace=None) -> bool:
        actual = str(_get_attr(prediction, 'response')).strip().lower()
        return all(kw.lower() in actual for kw in keywords)
    
    metric.__name__ = f"keyword_match_{len(keywords)}_keywords"
    return metric


def combined_metric(*metrics, weights: Optional[list] = None) -> Callable:
    """
    Combine multiple metrics with optional weights.
    
    Usage:
        metric = combined_metric(
            format_compliance("ANSWER:"),
            numeric_accuracy,
            weights=[0.3, 0.7]
        )
    """
    if weights is None:
        weights = [1.0 / len(metrics)] * len(metrics)
    
    if len(weights) != len(metrics):
        raise ValueError("Number of weights must match number of metrics")
    
    def metric(example, prediction, trace=None) -> float:
        total = 0.0
        for m, w in zip(metrics, weights):
            score = m(example, prediction, trace)
            total += float(score) * w
        return total
    
    metric.__name__ = "combined_metric"
    return metric


def create_metric(
    check_format: Optional[str] = None,
    check_contains: Optional[str] = None,
    check_numeric: bool = False,
    check_keywords: Optional[list] = None,
    tolerance: float = 0.01,
) -> Callable:
    """
    Convenience factory for creating custom metrics.
    
    Args:
        check_format: Required prefix (e.g., "ANSWER:")
        check_contains: Required substring
        check_numeric: Whether to check numeric accuracy
        check_keywords: Required keywords
        tolerance: Numeric tolerance
        
    Usage:
        metric = create_metric(
            check_format="RESULT:",
            check_numeric=True,
            tolerance=0.1
        )
    """
    checks = []
    
    if check_format:
        checks.append(format_compliance(check_format))
    
    if check_contains:
        def contains_check(example, prediction, trace=None):
            actual = str(_get_attr(prediction, 'response')).lower()
            return check_contains.lower() in actual
        checks.append(contains_check)
    
    if check_numeric:
        def numeric_check(example, prediction, trace=None):
            return numeric_accuracy(example, prediction, trace, tolerance)
        checks.append(numeric_check)
    
    if check_keywords:
        checks.append(keyword_match(check_keywords))
    
    if not checks:
        return exact_match
    
    if len(checks) == 1:
        return checks[0]
    
    def combined(example, prediction, trace=None) -> bool:
        return all(check(example, prediction, trace) for check in checks)
    
    combined.__name__ = "custom_metric"
    return combined