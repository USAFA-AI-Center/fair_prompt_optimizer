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
    Detects ANY attempt at JSON or ReAct formatting that leaked through.

    When fair_llm's planner fails to parse the model's response, it falls back to
    returning the raw text as a FinalAnswer. This metric aggressively detects those
    cases by looking for any structural fragments indicating the model attempted
    (but failed) to output the expected format.

    Returns True if the output appears to be a clean final answer.
    Returns False if the output contains format fragments (indicating parse failure).

    Detects:
    - Malformed JSON (ReActPlanner format failures)
    - JSON without proper quoting ({thought: instead of {"thought":)
    - Malformed key-value (SimpleReActPlanner format failures)
    - Truncated or unbalanced structures
    - Markdown code blocks containing JSON
    - Model announcing JSON output
    - Common typos in ReAct keys
    """
    response = str(_get_attr(prediction, 'response')).strip()
    response_lower = response.lower()

    # === JSON Format Fragments (ReActPlanner) ===
    # Properly quoted JSON keys
    json_fragments = [
        '{"thought"',
        '{"action"',
        '"tool_name"',
        '"tool_input"',
        '{"thought":',
        '{"action":',
        '"final_answer"',
        '"observation"',
    ]

    for fragment in json_fragments:
        if fragment in response:
            return False

    # === JSON without proper quotes (common LLM error) ===
    # Models sometimes output {thought: instead of {"thought":
    unquoted_json = [
        '{thought:',
        '{action:',
        '{tool_name:',
        '{tool_input:',
        '{ thought:',
        '{ action:',
    ]

    for fragment in unquoted_json:
        if fragment in response_lower:
            return False

    # === Common typos in ReAct keys ===
    typo_patterns = [
        '"thougth"',   # thought typo
        '"thougt"',    # thought typo
        '"thuoght"',   # thought typo
        '"acton"',     # action typo
        '"actoin"',    # action typo
        '"tool_nme"',  # tool_name typo
        '"toolname"',  # tool_name missing underscore
        '"toolinput"', # tool_input missing underscore
    ]

    for typo in typo_patterns:
        if typo in response_lower:
            return False

    # === Markdown code blocks with JSON ===
    if '```json' in response_lower or '```\n{' in response:
        return False

    # === Model announcing JSON output ===
    json_announcements = [
        'here is the json',
        'here\'s the json',
        'json response:',
        'json output:',
        'returning json',
        'my response in json',
        'formatted as json',
    ]

    for announcement in json_announcements:
        if announcement in response_lower:
            return False

    # === Truncated JSON ===
    # Starts with { but doesn't close properly
    if response.startswith('{') and not response.endswith('}'):
        return False

    # JSON-like structure with ReAct keywords
    if response.startswith('{') and 'thought' in response_lower:
        return False

    # Starts with { and contains action-related content
    if response.startswith('{') and ('action' in response_lower or 'tool' in response_lower):
        return False

    # === Key-Value Format Fragments (SimpleReActPlanner) ===
    kv_patterns = [
        'thought:',
        'action:',
        'tool_name:',
        'tool_input:',
        'observation:',
        'final_answer:',
    ]

    # Count how many KV patterns appear - if multiple, it's likely a leaked format
    kv_count = sum(1 for pattern in kv_patterns if pattern in response_lower)
    if kv_count >= 2:
        return False

    # Single "Action:" or "Thought:" at start of response is suspicious
    if response_lower.startswith('thought:') or response_lower.startswith('action:'):
        return False

    # === Unbalanced JSON Structures ===
    # Check for unbalanced braces (more opens than closes)
    if response.count('{') > response.count('}'):
        return False
    if response.count('[') > response.count(']'):
        return False

    # === Response starts with JSON-like structure ===
    # Even balanced JSON at start indicates format attempt
    if response.startswith('{') and response.count('{') >= 1:
        # Allow simple cases like "{name}" in natural text, but not JSON objects
        if ':' in response.split('}')[0]:  # Has key-value structure
            return False

    return True


def format_compliance_score(example, prediction, trace=None) -> float:
    """
    Graduated scoring version of json_format_compliance.

    Aggressively penalizes any attempt at JSON/ReAct formatting that failed parsing.

    Returns:
        1.0 - Clean final answer, no format leakage
        0.7 - Minor issues (single suspicious pattern)
        0.3 - Moderate issues (multiple format fragments detected)
        0.0 - Severe format failure (clearly malformed structure)
    """
    response = str(_get_attr(prediction, 'response')).strip()
    response_lower = response.lower()

    issues = 0
    severe = False

    # === Properly quoted JSON fragments (most severe) ===
    json_fragments = [
        '{"thought"', '{"action"', '"tool_name"', '"tool_input"',
        '"final_answer"', '"observation"'
    ]
    for fragment in json_fragments:
        if fragment in response:
            issues += 3
            severe = True

    # === Unquoted JSON attempts ===
    unquoted_json = ['{thought:', '{action:', '{tool_name:', '{tool_input:']
    for fragment in unquoted_json:
        if fragment in response_lower:
            issues += 2
            severe = True

    # === Common typos (indicates formatting attempt) ===
    typo_patterns = ['"thougth"', '"thougt"', '"acton"', '"toolname"']
    for typo in typo_patterns:
        if typo in response_lower:
            issues += 2
            severe = True

    # === Markdown code blocks ===
    if '```json' in response_lower or '```\n{' in response:
        issues += 2
        severe = True

    # === Model announcing JSON ===
    json_announcements = [
        'here is the json', 'here\'s the json', 'json response:',
        'json output:', 'returning json'
    ]
    for announcement in json_announcements:
        if announcement in response_lower:
            issues += 2

    # === KV patterns ===
    kv_patterns = [
        'thought:', 'action:', 'tool_name:', 'tool_input:',
        'observation:', 'final_answer:'
    ]
    kv_count = sum(1 for p in kv_patterns if p in response_lower)
    issues += kv_count

    # === Structural issues ===
    if response.startswith('{') and not response.endswith('}'):
        issues += 2
        severe = True

    if response.count('{') > response.count('}'):
        issues += 1

    # Response starts with JSON-like structure
    if response.startswith('{') and response.count('{') >= 1:
        if ':' in response.split('}')[0]:  # Has key-value structure
            issues += 2

    # Starts with { and contains ReAct keywords
    if response.startswith('{') and ('thought' in response_lower or 'action' in response_lower):
        issues += 2

    if severe or issues >= 4:
        return 0.0
    elif issues >= 2:
        return 0.3
    elif issues >= 1:
        return 0.7
    else:
        return 1.0

# Custom metric for research quality
def research_quality_metric(example, prediction, trace=None) -> float:
    """
    Evaluate research output quality.
    
    Checks for:
    - Presence of substantive content (not just errors)
    - Mentions of sources/data
    - Coherent structure
    
    Returns:
        1.0 - High quality research output
        0.5 - Partial/incomplete output
        0.0 - Failed or empty output
    """
    response = str(prediction.response).lower() if hasattr(prediction, 'response') else ""
    
    # Check for failure indicators
    if not response or "error" in response or len(response) < 50:
        return 0.0
    
    score = 0.0
    
    # Has substantive content
    if len(response) > 200:
        score += 0.3
    
    # Mentions sources or data
    source_indicators = ["according to", "research", "found", "shows", "indicates", "source", "study"]
    if any(indicator in response for indicator in source_indicators):
        score += 0.3
    
    # Has structure (multiple points/sentences)
    if response.count(". ") >= 2:
        score += 0.2
    
    # Addresses the query topic (check if expected_output keywords appear)
    expected = str(example.expected_output).lower() if hasattr(example, 'expected_output') else ""
    if expected:
        expected_keywords = [w for w in expected.split() if len(w) > 4][:5]
        matches = sum(1 for kw in expected_keywords if kw in response)
        score += 0.2 * (matches / max(len(expected_keywords), 1))
    else:
        score += 0.2  # No expected output to compare, give benefit of doubt
    
    return min(score, 1.0)


def numeric_accuracy_with_format(example, prediction, trace=None, tolerance: float = 0.00) -> float:
    """
    Combined metric: answer must be correct AND properly formatted.

    Detects when the model gets the right answer but through format fallback
    (malformed JSON/KV that fair_llm's planner couldn't parse, so it returned
    the raw text as FinalAnswer).

    Returns:
        1.0 - Correct answer via clean format
        0.5 - Correct answer but format leaked (got lucky with fallback parsing)
        0.0 - Wrong answer
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