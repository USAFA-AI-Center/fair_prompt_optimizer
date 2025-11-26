"""
metrics.py
==========

Common evaluation metrics for prompt optimization.

These metrics are designed to work with DSPy's optimization loop and can be
used directly with FAIRPromptOptimizer.optimize_*() methods.

All metrics follow the signature:
    metric(example, prediction, trace=None) -> float | bool
    
Where:
    - example: The input example with expected output
    - prediction: The model's prediction  
    - trace: Optional trace information (usually None)
    
Returns a score between 0.0 and 1.0, or a boolean.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def exact_match(example, prediction, trace=None) -> bool:
    """
    Check if the prediction exactly matches the expected output.
    
    Performs case-insensitive comparison after stripping whitespace.
    
    Args:
        example: Example with 'response' or output field
        prediction: Prediction with 'response' field
        
    Returns:
        True if exact match, False otherwise
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return False
    
    return expected.strip().lower() == predicted.strip().lower()


def exact_match_strict(example, prediction, trace=None) -> bool:
    """
    Strict exact match - case sensitive, no whitespace normalization.
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return False
    
    return expected == predicted


def contains_answer(example, prediction, trace=None) -> bool:
    """
    Check if the expected answer appears anywhere in the prediction.
    
    Useful when the model provides verbose responses that contain
    the correct answer among other text.
    
    Args:
        example: Example with expected output
        prediction: Model's prediction
        
    Returns:
        True if expected is contained in predicted
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return False
    
    return expected.strip().lower() in predicted.strip().lower()


def contains_all_keywords(example, prediction, trace=None) -> float:
    """
    Check what fraction of expected keywords appear in the prediction.
    
    Splits expected output into words and checks how many appear
    in the prediction.
    
    Returns:
        Float between 0.0 and 1.0 representing keyword coverage
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return 0.0
    
    # Extract keywords (words with 3+ characters)
    expected_words = set(
        word.lower() for word in re.findall(r'\b\w{3,}\b', expected)
    )
    
    if not expected_words:
        return 1.0  # No keywords to match
    
    predicted_lower = predicted.lower()
    matches = sum(1 for word in expected_words if word in predicted_lower)
    
    return matches / len(expected_words)


def numeric_accuracy(example, prediction, trace=None, tolerance: float = 0.01) -> bool:
    """
    Check if numeric answers match within a tolerance.
    
    Extracts the last number from both expected and predicted outputs
    and compares them.
    
    Args:
        example: Example with expected numeric output
        prediction: Model's prediction
        tolerance: Allowed relative difference (default 1%)
        
    Returns:
        True if numbers match within tolerance
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return False
    
    expected_num = _extract_number(expected)
    predicted_num = _extract_number(predicted)
    
    if expected_num is None or predicted_num is None:
        return False
    
    # Handle zero case
    if expected_num == 0:
        return abs(predicted_num) < tolerance
    
    # Relative difference
    rel_diff = abs(expected_num - predicted_num) / abs(expected_num)
    return rel_diff <= tolerance


def numeric_accuracy_absolute(
    example, prediction, trace=None, tolerance: float = 0.5
) -> bool:
    """
    Check if numeric answers match within an absolute tolerance.
    
    Args:
        example: Example with expected numeric output
        prediction: Model's prediction  
        tolerance: Allowed absolute difference
        
    Returns:
        True if numbers are within tolerance
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return False
    
    expected_num = _extract_number(expected)
    predicted_num = _extract_number(predicted)
    
    if expected_num is None or predicted_num is None:
        return False
    
    return abs(expected_num - predicted_num) <= tolerance



def fuzzy_match(example, prediction, trace=None, threshold: float = 0.8) -> float:
    """
    Compute fuzzy string similarity using Levenshtein ratio.
    
    Args:
        example: Example with expected output
        prediction: Model's prediction
        threshold: Minimum ratio to consider a match (for boolean mode)
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return 0.0
    
    # Normalize strings
    expected = expected.strip().lower()
    predicted = predicted.strip().lower()
    
    if not expected or not predicted:
        return 0.0 if expected != predicted else 1.0
    
    # Simple similarity based on common subsequence
    return _simple_ratio(expected, predicted)


def jaccard_similarity(example, prediction, trace=None) -> float:
    """
    Compute Jaccard similarity between word sets.
    
    Args:
        example: Example with expected output
        prediction: Model's prediction
        
    Returns:
        Jaccard index between 0.0 and 1.0
    """
    expected = _get_expected(example)
    predicted = _get_predicted(prediction)
    
    if expected is None or predicted is None:
        return 0.0
    
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())
    
    if not expected_words and not predicted_words:
        return 1.0
    
    if not expected_words or not predicted_words:
        return 0.0
    
    intersection = expected_words & predicted_words
    union = expected_words | predicted_words
    
    return len(intersection) / len(union)


def combined_metric(
    example, prediction, trace=None,
    exact_weight: float = 0.5,
    contains_weight: float = 0.3,
    fuzzy_weight: float = 0.2
) -> float:
    """
    Combine multiple metrics with configurable weights.
    
    Args:
        example: Example with expected output
        prediction: Model's prediction
        exact_weight: Weight for exact match
        contains_weight: Weight for containment
        fuzzy_weight: Weight for fuzzy match
        
    Returns:
        Weighted combination of metric scores
    """
    exact = 1.0 if exact_match(example, prediction, trace) else 0.0
    contains = 1.0 if contains_answer(example, prediction, trace) else 0.0
    fuzzy = fuzzy_match(example, prediction, trace)
    
    total_weight = exact_weight + contains_weight + fuzzy_weight
    
    return (
        exact * exact_weight +
        contains * contains_weight +
        fuzzy * fuzzy_weight
    ) / total_weight


def create_numeric_metric(tolerance: float = 0.01, absolute: bool = False):
    """
    Factory to create a numeric accuracy metric with custom tolerance.
    
    Args:
        tolerance: Allowed difference
        absolute: Use absolute (True) or relative (False) tolerance
        
    Returns:
        Configured metric function
    """
    def metric(example, prediction, trace=None):
        if absolute:
            return numeric_accuracy_absolute(example, prediction, trace, tolerance)
        return numeric_accuracy(example, prediction, trace, tolerance)
    
    return metric


def create_fuzzy_metric(threshold: float = 0.8):
    """
    Factory to create a fuzzy match metric with custom threshold.
    
    Args:
        threshold: Minimum similarity threshold
        
    Returns:
        Configured metric function returning bool
    """
    def metric(example, prediction, trace=None):
        score = fuzzy_match(example, prediction, trace)
        return score >= threshold
    
    return metric


def create_combined_metric(
    exact_weight: float = 0.5,
    contains_weight: float = 0.3,
    fuzzy_weight: float = 0.2
):
    """
    Factory to create a combined metric with custom weights.
    
    Returns:
        Configured combined metric function
    """
    def metric(example, prediction, trace=None):
        return combined_metric(
            example, prediction, trace,
            exact_weight, contains_weight, fuzzy_weight
        )
    
    return metric


def _get_expected(example) -> Optional[str]:
    """Extract expected output from an example."""
    if hasattr(example, 'response'):
        return str(example.response)
    if hasattr(example, 'expected_output'):
        return str(example.expected_output)
    if hasattr(example, 'answer'):
        return str(example.answer)
    if hasattr(example, 'output'):
        return str(example.output)
    if isinstance(example, dict):
        for key in ['response', 'expected_output', 'answer', 'output']:
            if key in example:
                return str(example[key])
    return None


def _get_predicted(prediction) -> Optional[str]:
    """Extract predicted output from a prediction."""
    if hasattr(prediction, 'response'):
        return str(prediction.response)
    if hasattr(prediction, 'answer'):
        return str(prediction.answer)
    if hasattr(prediction, 'output'):
        return str(prediction.output)
    if isinstance(prediction, dict):
        for key in ['response', 'answer', 'output']:
            if key in prediction:
                return str(prediction[key])
    if isinstance(prediction, str):
        return prediction
    return None


def _extract_number(text: str) -> Optional[float]:
    """Extract the last number from a string."""
    # Find all numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if not numbers:
        return None
    
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _simple_ratio(s1: str, s2: str) -> float:
    """
    Compute a simple similarity ratio between two strings.
    
    Uses longest common subsequence approach.
    """
    if not s1 or not s2:
        return 0.0
    
    # Length of longest common subsequence
    m, n = len(s1), len(s2)
    
    # Use shorter computation for long strings
    if m > 100 or n > 100:
        # Fall back to word-based comparison for long strings
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        return 2.0 * intersection / (len(words1) + len(words2))
    
    # Dynamic programming for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    return 2.0 * lcs_length / (m + n)
