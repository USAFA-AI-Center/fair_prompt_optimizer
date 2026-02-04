# tests/test_metrics.py
"""
Unit tests for fair_prompt_optimizer metrics module.

Run with: pytest tests/test_metrics.py -v
"""

import pytest
from dataclasses import dataclass
from typing import Any


@dataclass
class MockExample:
    """Mock training example."""
    expected_output: str
    inputs: dict = None


@dataclass  
class MockPrediction:
    """Mock prediction from model."""
    response: str


class TestExactMatch:
    """Test exact_match metric."""
    
    def test_exact_match_true(self):
        from fair_prompt_optimizer.metrics import exact_match
        
        example = MockExample(expected_output="Hello")
        prediction = MockPrediction(response="Hello")
        
        assert exact_match(example, prediction) == True
    
    def test_exact_match_false(self):
        from fair_prompt_optimizer.metrics import exact_match
        
        example = MockExample(expected_output="Hello")
        prediction = MockPrediction(response="Hi")
        
        assert exact_match(example, prediction) == False
    
    def test_exact_match_whitespace(self):
        from fair_prompt_optimizer.metrics import exact_match
        
        example = MockExample(expected_output="  Hello  ")
        prediction = MockPrediction(response="Hello")
        
        assert exact_match(example, prediction) == True


class TestContainsAnswer:
    """Test contains_answer metric."""
    
    def test_contains_true(self):
        from fair_prompt_optimizer.metrics import contains_answer
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="The answer is 42.")
        
        assert contains_answer(example, prediction) == True
    
    def test_contains_false(self):
        from fair_prompt_optimizer.metrics import contains_answer
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="The answer is 43.")
        
        assert contains_answer(example, prediction) == False
    
    def test_contains_case_insensitive(self):
        from fair_prompt_optimizer.metrics import contains_answer
        
        example = MockExample(expected_output="HELLO")
        prediction = MockPrediction(response="hello world")
        
        assert contains_answer(example, prediction) == True


class TestNumericAccuracy:
    """Test numeric_accuracy metric."""
    
    def test_exact_number(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="42")
        
        assert numeric_accuracy(example, prediction) == True
    
    def test_number_in_text(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="The result is 42.")
        
        assert numeric_accuracy(example, prediction) == True
    
    def test_tolerance(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        example = MockExample(expected_output="42.0")
        prediction = MockPrediction(response="42.005")
        
        assert numeric_accuracy(example, prediction, tolerance=0.01) == True
        assert numeric_accuracy(example, prediction, tolerance=0.001) == False
    
    def test_decimal_numbers(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        example = MockExample(expected_output="3.14159")
        prediction = MockPrediction(response="Pi is approximately 3.14159")
        
        assert numeric_accuracy(example, prediction) == True
    
    def test_no_number(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy
        
        example = MockExample(expected_output="hello")
        prediction = MockPrediction(response="world")
        
        assert numeric_accuracy(example, prediction) == False


class TestFormatCompliance:
    """Test format_compliance metric factory."""
    
    def test_prefix_match(self):
        from fair_prompt_optimizer.metrics import format_compliance
        
        metric = format_compliance("ANSWER:")
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="ANSWER: 42")
        
        assert metric(example, prediction) == True
    
    def test_prefix_no_match(self):
        from fair_prompt_optimizer.metrics import format_compliance
        
        metric = format_compliance("ANSWER:")
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The answer is 42")
        
        assert metric(example, prediction) == False
    
    def test_prefix_case_insensitive(self):
        from fair_prompt_optimizer.metrics import format_compliance
        
        metric = format_compliance("ANSWER:")
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="answer: 42")
        
        assert metric(example, prediction) == True


class TestFuzzyMatch:
    """Test fuzzy_match metric."""
    
    def test_identical(self):
        from fair_prompt_optimizer.metrics import fuzzy_match
        
        example = MockExample(expected_output="hello")
        prediction = MockPrediction(response="hello")
        
        assert fuzzy_match(example, prediction) == 1.0
    
    def test_partial(self):
        from fair_prompt_optimizer.metrics import fuzzy_match
        
        # Use strings with different character sets for partial match
        example = MockExample(expected_output="hello")
        prediction = MockPrediction(response="help")  # shares h, e, l but has p instead of o
        
        score = fuzzy_match(example, prediction)
        assert 0.5 < score < 1.0  # Jaccard on {h,e,l,o} vs {h,e,l,p} = 3/5 = 0.6
    
    def test_completely_different(self):
        from fair_prompt_optimizer.metrics import fuzzy_match
        
        example = MockExample(expected_output="abc")
        prediction = MockPrediction(response="xyz")
        
        score = fuzzy_match(example, prediction)
        assert score == 0.0


class TestKeywordMatch:
    """Test keyword_match metric factory."""
    
    def test_all_keywords_present(self):
        from fair_prompt_optimizer.metrics import keyword_match
        
        metric = keyword_match(["price", "total", "$"])
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The total price is $50")
        
        assert metric(example, prediction) == True
    
    def test_missing_keyword(self):
        from fair_prompt_optimizer.metrics import keyword_match
        
        metric = keyword_match(["price", "total", "$"])
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The price is 50")
        
        assert metric(example, prediction) == False


class TestCombinedMetric:
    """Test combined_metric factory."""
    
    def test_all_pass(self):
        from fair_prompt_optimizer.metrics import combined_metric, format_compliance, contains_answer
        
        metric = combined_metric(
            format_compliance("RESULT:"),
            contains_answer,
        )
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="RESULT: The answer is 42")
        
        score = metric(example, prediction)
        assert score == 1.0
    
    def test_partial_pass(self):
        from fair_prompt_optimizer.metrics import combined_metric, format_compliance, contains_answer
        
        metric = combined_metric(
            format_compliance("RESULT:"),
            contains_answer,
            weights=[0.5, 0.5]
        )
        
        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="The answer is 42")  # No RESULT: prefix
        
        score = metric(example, prediction)
        assert score == 0.5


class TestCreateMetric:
    """Test create_metric factory."""
    
    def test_single_check(self):
        from fair_prompt_optimizer.metrics import create_metric
        
        metric = create_metric(check_format="ANSWER:")
        
        example = MockExample(expected_output="")
        prediction = MockPrediction(response="ANSWER: 42")
        
        assert metric(example, prediction) == True
    
    def test_multiple_checks(self):
        from fair_prompt_optimizer.metrics import create_metric
        
        metric = create_metric(
            check_format="RESULT:",
            check_numeric=True,
        )
        
        example = MockExample(expected_output="42")
        good_prediction = MockPrediction(response="RESULT: 42")
        bad_prediction = MockPrediction(response="42")  # Missing format
        
        assert metric(example, good_prediction) == True
        assert metric(example, bad_prediction) == False
    
    def test_empty_returns_exact_match(self):
        from fair_prompt_optimizer.metrics import create_metric, exact_match
        
        metric = create_metric()
        
        example = MockExample(expected_output="hello")
        prediction = MockPrediction(response="hello")
        
        assert metric(example, prediction) == exact_match(example, prediction)


class TestJsonFormatCompliance:
    """Test json_format_compliance metric."""

    def test_clean_answer_passes(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="42")

        assert json_format_compliance(example, prediction) == True

    def test_clean_natural_language_passes(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The result of the calculation is 42.")

        assert json_format_compliance(example, prediction) == True

    def test_json_fragment_fails(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response='{"thought": "I calculated", "action": {...}}')

        assert json_format_compliance(example, prediction) == False

    def test_truncated_json_fails(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response='{"thought": "incomplete')

        assert json_format_compliance(example, prediction) == False

    def test_react_response_leaked_fails(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response='{"thought": "My thought here", "answer": 42}')

        assert json_format_compliance(example, prediction) == False

    def test_tool_name_fragment_fails(self):
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="result")
        prediction = MockPrediction(response='"tool_name": "calculator", "tool_input": "5+5"')

        assert json_format_compliance(example, prediction) == False

    def test_kv_format_multiple_patterns_fails(self):
        """Test that SimpleReActPlanner KV format leakage is detected."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="42")
        # This is what a leaked SimpleReActPlanner response looks like
        prediction = MockPrediction(
            response="Thought: I need to calculate this\nAction:\ntool_name: calculator\ntool_input: 5+5"
        )

        assert json_format_compliance(example, prediction) == False

    def test_kv_format_starts_with_thought_fails(self):
        """Response starting with 'Thought:' is suspicious."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="Thought: The answer appears to be 42")

        assert json_format_compliance(example, prediction) == False

    def test_kv_format_starts_with_action_fails(self):
        """Response starting with 'Action:' is suspicious."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="Action: Using the calculator tool")

        assert json_format_compliance(example, prediction) == False

    def test_unbalanced_braces_fails(self):
        """Unbalanced braces indicate truncated JSON."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="Here is some data: { key: value, nested: { inner")

        assert json_format_compliance(example, prediction) == False

    def test_unbalanced_brackets_fails(self):
        """Unbalanced brackets indicate truncated structure."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="Array: [1, 2, 3, [4, 5")

        assert json_format_compliance(example, prediction) == False

    def test_final_answer_fragment_fails(self):
        """Detection of 'final_answer' in quotes indicates JSON leak."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='"final_answer": "The result is 42"')

        assert json_format_compliance(example, prediction) == False

    def test_single_kv_pattern_passes(self):
        """Single KV pattern word in natural text should pass."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        # The word "action" in natural context shouldn't fail
        prediction = MockPrediction(response="The recommended action is to proceed carefully.")

        # This passes because it doesn't start with "action:" and only has 1 pattern
        assert json_format_compliance(example, prediction) == True

    def test_unquoted_json_fails(self):
        """JSON without proper quotes should fail."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{thought: "I need to calculate", action: {...}}')

        assert json_format_compliance(example, prediction) == False

    def test_typo_in_react_key_fails(self):
        """Common typos in ReAct keys should be detected."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{"thougth": "calculating...", "action": {}}')

        assert json_format_compliance(example, prediction) == False

    def test_markdown_json_block_fails(self):
        """Markdown code blocks with JSON should fail."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='```json\n{"thought": "test"}\n```')

        assert json_format_compliance(example, prediction) == False

    def test_json_announcement_fails(self):
        """Model announcing JSON output should fail."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='Here is the JSON response: {"result": 42}')

        assert json_format_compliance(example, prediction) == False

    def test_observation_pattern_fails(self):
        """Observation: pattern should fail (ReAct component)."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='Thought: analyzing\nObservation: the result is 42')

        assert json_format_compliance(example, prediction) == False

    def test_json_starting_with_tool_fails(self):
        """JSON starting with { containing tool reference should fail."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{"name": "calculator", "input": "5+5"}')

        assert json_format_compliance(example, prediction) == False

    def test_balanced_json_with_colon_fails(self):
        """Balanced JSON-like structure with key:value should fail."""
        from fair_prompt_optimizer.metrics import json_format_compliance

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{result: 42}')

        assert json_format_compliance(example, prediction) == False


class TestFormatComplianceScore:
    """Test format_compliance_score graduated metric."""

    def test_clean_answer_full_score(self):
        from fair_prompt_optimizer.metrics import format_compliance_score

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The answer is 42.")

        assert format_compliance_score(example, prediction) == 1.0

    def test_severe_json_leak_zero_score(self):
        from fair_prompt_optimizer.metrics import format_compliance_score

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{"thought": "calculating", "action": {"tool_name": "calc"}}')

        assert format_compliance_score(example, prediction) == 0.0

    def test_truncated_json_zero_score(self):
        from fair_prompt_optimizer.metrics import format_compliance_score

        example = MockExample(expected_output="")
        prediction = MockPrediction(response='{"thought": "incomplete')

        assert format_compliance_score(example, prediction) == 0.0

    def test_moderate_kv_leak_low_score(self):
        from fair_prompt_optimizer.metrics import format_compliance_score

        example = MockExample(expected_output="")
        # Multiple KV patterns but no severe JSON fragments
        prediction = MockPrediction(response="Thought: thinking\ntool_name: calc")

        score = format_compliance_score(example, prediction)
        assert score == 0.3  # Moderate issues

    def test_minor_issue_partial_score(self):
        from fair_prompt_optimizer.metrics import format_compliance_score

        example = MockExample(expected_output="")
        # Just one unbalanced brace
        prediction = MockPrediction(response="The data shows { partial info")

        score = format_compliance_score(example, prediction)
        assert score == 0.7  # Minor issue


class TestNumericAccuracyWithFormat:
    """Test numeric_accuracy_with_format metric."""

    def test_correct_answer_clean_format(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy_with_format

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="42")

        assert numeric_accuracy_with_format(example, prediction) == 1.0

    def test_correct_answer_bad_format(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy_with_format

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response='{"thought": "calculated"} 42')

        assert numeric_accuracy_with_format(example, prediction) == 0.5

    def test_wrong_answer(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy_with_format

        example = MockExample(expected_output="42")
        prediction = MockPrediction(response="43")

        assert numeric_accuracy_with_format(example, prediction) == 0.0


class TestSentimentFormatMetric:
    """Test sentiment_format_metric."""

    def test_perfect_format_positive(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="SENTIMENT: positive")

        assert sentiment_format_metric(example, prediction) == 1.0

    def test_perfect_format_negative(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="sentiment: negative")

        assert sentiment_format_metric(example, prediction) == 1.0

    def test_perfect_format_neutral(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="SENTIMENT: neutral")

        assert sentiment_format_metric(example, prediction) == 1.0

    def test_sentiment_prefix_partial(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="sentiment: maybe positive")

        assert sentiment_format_metric(example, prediction) == 0.8

    def test_sentiment_somewhere_in_text(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="The sentiment is clearly positive.")

        assert sentiment_format_metric(example, prediction) == 0.5

    def test_just_label(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="positive")

        assert sentiment_format_metric(example, prediction) == 0.3

    def test_label_in_short_text(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="It is positive")

        assert sentiment_format_metric(example, prediction) == 0.2

    def test_garbage_output(self):
        from fair_prompt_optimizer.metrics import sentiment_format_metric

        example = MockExample(expected_output="")
        prediction = MockPrediction(response="I don't understand the question")

        assert sentiment_format_metric(example, prediction) == 0.0


class TestResearchQualityMetric:
    """Test research_quality_metric."""

    def test_high_quality_output(self):
        from fair_prompt_optimizer.metrics import research_quality_metric

        example = MockExample(expected_output="climate change effects")

        long_response = (
            "According to recent research, climate change has significant effects on agriculture. "
            "Studies show that changing temperature patterns affect crop yields. "
            "The research indicates that farmers need to adapt their practices. "
            "Multiple sources confirm these findings from various scientific organizations."
        )
        prediction = MockPrediction(response=long_response)

        score = research_quality_metric(example, prediction)
        assert score >= 0.7  # Should be high quality

    def test_empty_output(self):
        from fair_prompt_optimizer.metrics import research_quality_metric

        example = MockExample(expected_output="test")
        prediction = MockPrediction(response="")

        assert research_quality_metric(example, prediction) == 0.0

    def test_error_output(self):
        from fair_prompt_optimizer.metrics import research_quality_metric

        example = MockExample(expected_output="test")
        prediction = MockPrediction(response="Error: could not process request")

        assert research_quality_metric(example, prediction) == 0.0

    def test_short_output(self):
        from fair_prompt_optimizer.metrics import research_quality_metric

        example = MockExample(expected_output="test")
        prediction = MockPrediction(response="Short")

        assert research_quality_metric(example, prediction) == 0.0

    def test_medium_quality_output(self):
        from fair_prompt_optimizer.metrics import research_quality_metric

        example = MockExample(expected_output="")

        medium_response = (
            "The topic has several aspects. Research shows some findings. "
            "There are multiple perspectives to consider."
        )
        prediction = MockPrediction(response=medium_response)

        score = research_quality_metric(example, prediction)
        assert 0.2 < score < 0.8  # Medium quality


class TestGetAttrHelper:
    """Test _get_attr helper function."""

    def test_get_attr_from_object(self):
        from fair_prompt_optimizer.metrics import _get_attr

        obj = MockExample(expected_output="test")
        assert _get_attr(obj, "expected_output") == "test"

    def test_get_attr_from_dict(self):
        from fair_prompt_optimizer.metrics import _get_attr

        obj = {"expected_output": "test_value", "other": "data"}
        assert _get_attr(obj, "expected_output") == "test_value"

    def test_get_attr_missing_returns_default(self):
        from fair_prompt_optimizer.metrics import _get_attr

        obj = MockExample(expected_output="test")
        assert _get_attr(obj, "nonexistent", "default") == "default"

    def test_get_attr_dict_missing_returns_default(self):
        from fair_prompt_optimizer.metrics import _get_attr

        obj = {"key": "value"}
        assert _get_attr(obj, "missing", "fallback") == "fallback"


class TestCombinedMetricWeights:
    """Test combined_metric with various weight configurations."""

    def test_unequal_weights(self):
        from fair_prompt_optimizer.metrics import combined_metric, exact_match, contains_answer

        # Weight exact_match heavily
        metric = combined_metric(
            exact_match,
            contains_answer,
            weights=[0.8, 0.2]
        )

        example = MockExample(expected_output="42")
        # Exact match fails, contains passes
        prediction = MockPrediction(response="The answer is 42")

        score = metric(example, prediction)
        assert score == pytest.approx(0.2)  # Only contains_answer passes

    def test_weight_validation(self):
        from fair_prompt_optimizer.metrics import combined_metric, exact_match, contains_answer

        with pytest.raises(ValueError, match="weights must match"):
            combined_metric(
                exact_match,
                contains_answer,
                weights=[0.5]  # Wrong number of weights
            )


class TestCreateMetricWithContains:
    """Test create_metric with check_contains option."""

    def test_check_contains(self):
        from fair_prompt_optimizer.metrics import create_metric

        metric = create_metric(check_contains="success")

        example = MockExample(expected_output="")
        good = MockPrediction(response="Operation was a success!")
        bad = MockPrediction(response="Operation failed")

        assert metric(example, good) == True
        assert metric(example, bad) == False


class TestNegativeNumbers:
    """Test numeric metrics with negative numbers."""

    def test_numeric_accuracy_negative(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy

        example = MockExample(expected_output="-42")
        prediction = MockPrediction(response="The result is -42")

        assert numeric_accuracy(example, prediction) == True

    def test_numeric_accuracy_negative_tolerance(self):
        from fair_prompt_optimizer.metrics import numeric_accuracy

        example = MockExample(expected_output="-10.0")
        prediction = MockPrediction(response="-10.05")

        assert numeric_accuracy(example, prediction, tolerance=0.1) == True
        assert numeric_accuracy(example, prediction, tolerance=0.01) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])