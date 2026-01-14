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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])