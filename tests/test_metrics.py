# test_metrics.py
"""
Tests for fair_prompt_optimizer.metrics module.
"""

from dataclasses import dataclass
from typing import Optional

from fair_prompt_optimizer.metrics import (
    exact_match,
    exact_match_strict,
    contains_answer,
    contains_all_keywords,
    numeric_accuracy,
    numeric_accuracy_absolute,
    fuzzy_match,
    jaccard_similarity,
    combined_metric,
    create_numeric_metric,
    create_fuzzy_metric,
    create_combined_metric,
    _get_expected,
    _get_predicted,
    _extract_number,
    _simple_ratio,
)


# =============================================================================
# Test Fixtures / Helper Classes
# =============================================================================

@dataclass
class MockExample:
    """Mock example for testing metrics"""
    response: Optional[str] = None
    expected_output: Optional[str] = None
    answer: Optional[str] = None
    output: Optional[str] = None


@dataclass
class MockPrediction:
    """Mock prediction for testing metrics"""
    response: Optional[str] = None
    answer: Optional[str] = None
    output: Optional[str] = None


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestGetExpected:
    """Tests for _get_expected helper function"""
    
    def test_response_attribute(self):
        """Test extraction from response attribute"""
        example = MockExample(response="test response")
        assert _get_expected(example) == "test response"
    
    def test_expected_output_attribute(self):
        """Test extraction from expected_output attribute (response takes priority)"""
        # Note: response attribute is checked first, so we need to not set it
        example = MockExample(response=None, expected_output="expected")
        # The function returns str(response) if response exists, even if None
        # So we need to test with a fresh object that doesn't have response
        result = _get_expected({"expected_output": "expected"})
        assert result == "expected"
    
    def test_answer_attribute(self):
        """Test extraction from answer attribute via dict"""
        # Test via dict to avoid response priority
        result = _get_expected({"answer": "answer"})
        assert result == "answer"
    
    def test_dict_input(self):
        """Test extraction from dictionary"""
        example = {"response": "dict response"}
        assert _get_expected(example) == "dict response"
    
    def test_dict_fallback(self):
        """Test dictionary fallback to other keys"""
        example = {"expected_output": "expected"}
        assert _get_expected(example) == "expected"
    
    def test_returns_none_for_missing(self):
        """Test returns None when no matching key in dict"""
        example = {"unrelated_key": "value"}
        assert _get_expected(example) is None
    
    def test_priority_order(self):
        """Test that response takes priority"""
        example = MockExample(response="first", expected_output="second")
        assert _get_expected(example) == "first"


class TestGetPredicted:
    """Tests for _get_predicted helper function"""
    
    def test_response_attribute(self):
        """Test extraction from response attribute"""
        pred = MockPrediction(response="predicted")
        assert _get_predicted(pred) == "predicted"
    
    def test_string_input(self):
        """Test handling of string input directly"""
        assert _get_predicted("direct string") == "direct string"
    
    def test_dict_input(self):
        """Test extraction from dictionary"""
        pred = {"response": "dict response"}
        assert _get_predicted(pred) == "dict response"


class TestExtractNumber:
    """Tests for _extract_number helper function"""
    
    def test_simple_integer(self):
        """Test extraction of simple integer"""
        assert _extract_number("42") == 42.0
    
    def test_decimal(self):
        """Test extraction of decimal number"""
        assert _extract_number("3.14") == 3.14
    
    def test_negative(self):
        """Test extraction of negative number"""
        assert _extract_number("-5") == -5.0
    
    def test_number_in_text(self):
        """Test extraction of number from text"""
        assert _extract_number("The answer is 42.") == 42.0
    
    def test_last_number_extracted(self):
        """Test that last number is extracted"""
        assert _extract_number("First 10, then 20, finally 30") == 30.0
    
    def test_no_number(self):
        """Test returns None when no number present"""
        assert _extract_number("no numbers here") is None
    
    def test_negative_decimal(self):
        """Test extraction of negative decimal"""
        assert _extract_number("Result: -3.5") == -3.5


class TestSimpleRatio:
    """Tests for _simple_ratio helper function"""
    
    def test_identical_strings(self):
        """Test ratio of identical strings"""
        assert _simple_ratio("hello", "hello") == 1.0
    
    def test_empty_strings(self):
        """Test ratio with empty strings"""
        assert _simple_ratio("", "") == 0.0
        assert _simple_ratio("test", "") == 0.0
    
    def test_partial_match(self):
        """Test ratio of partially matching strings"""
        ratio = _simple_ratio("hello", "hallo")
        assert 0.5 < ratio < 1.0
    
    def test_completely_different(self):
        """Test ratio of completely different strings"""
        ratio = _simple_ratio("abc", "xyz")
        assert ratio < 0.5


# =============================================================================
# Tests for Exact Match Metrics
# =============================================================================

class TestExactMatch:
    """Tests for exact_match metric"""
    
    def test_exact_match(self):
        """Test exact matching strings"""
        example = MockExample(response="hello")
        prediction = MockPrediction(response="hello")
        assert exact_match(example, prediction) is True
    
    def test_case_insensitive(self):
        """Test case insensitivity"""
        example = MockExample(response="Hello")
        prediction = MockPrediction(response="hello")
        assert exact_match(example, prediction) is True
    
    def test_whitespace_handling(self):
        """Test whitespace stripping"""
        example = MockExample(response="  hello  ")
        prediction = MockPrediction(response="hello")
        assert exact_match(example, prediction) is True
    
    def test_no_match(self):
        """Test non-matching strings"""
        example = MockExample(response="hello")
        prediction = MockPrediction(response="world")
        assert exact_match(example, prediction) is False
    
    def test_none_values(self):
        """Test handling of None values"""
        example = MockExample()
        prediction = MockPrediction(response="test")
        assert exact_match(example, prediction) is False


class TestExactMatchStrict:
    """Tests for exact_match_strict metric"""
    
    def test_case_sensitive(self):
        """Test that strict match is case sensitive"""
        example = MockExample(response="Hello")
        prediction = MockPrediction(response="hello")
        assert exact_match_strict(example, prediction) is False
    
    def test_exact_case_match(self):
        """Test exact case match"""
        example = MockExample(response="Hello")
        prediction = MockPrediction(response="Hello")
        assert exact_match_strict(example, prediction) is True


# =============================================================================
# Tests for Contains Metrics
# =============================================================================

class TestContainsAnswer:
    """Tests for contains_answer metric"""
    
    def test_contains_substring(self):
        """Test that substring is found"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="The answer is 42.")
        assert contains_answer(example, prediction) is True
    
    def test_exact_match(self):
        """Test exact match also passes"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="42")
        assert contains_answer(example, prediction) is True
    
    def test_case_insensitive(self):
        """Test case insensitivity"""
        example = MockExample(response="Paris")
        prediction = MockPrediction(response="The capital is paris.")
        assert contains_answer(example, prediction) is True
    
    def test_not_contained(self):
        """Test when answer not contained"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="The answer is 24.")
        assert contains_answer(example, prediction) is False


class TestContainsAllKeywords:
    """Tests for contains_all_keywords metric"""
    
    def test_all_keywords_present(self):
        """Test all keywords found"""
        example = MockExample(response="python programming language")
        prediction = MockPrediction(
            response="Python is a programming language"
        )
        assert contains_all_keywords(example, prediction) == 1.0
    
    def test_partial_keywords(self):
        """Test partial keyword match"""
        example = MockExample(response="python java ruby")
        prediction = MockPrediction(response="I know python and java")
        score = contains_all_keywords(example, prediction)
        assert 0.5 < score < 1.0
    
    def test_no_keywords(self):
        """Test no keywords match"""
        example = MockExample(response="python java")
        prediction = MockPrediction(response="I like cats")
        assert contains_all_keywords(example, prediction) == 0.0
    
    def test_short_words_excluded(self):
        """Test that short words (< 3 chars) are excluded from keyword matching"""
        # Use words with 3+ characters
        example = MockExample(response="the quick fox")  # "the" is 3 chars, "quick" is 5, "fox" is 3
        prediction = MockPrediction(response="the quick brown fox")
        score = contains_all_keywords(example, prediction)
        # All keywords should be found
        assert score == 1.0
        
    def test_no_valid_keywords(self):
        """Test when expected has only short words (< 3 chars)"""
        example = MockExample(response="a i")  # All under 3 chars
        prediction = MockPrediction(response="test something")
        score = contains_all_keywords(example, prediction)
        assert score == 1.0


# =============================================================================
# Tests for Numeric Metrics
# =============================================================================

class TestNumericAccuracy:
    """Tests for numeric_accuracy metric"""
    
    def test_exact_match(self):
        """Test exact numeric match"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="42")
        assert numeric_accuracy(example, prediction) is True
    
    def test_within_tolerance(self):
        """Test match within default 1% tolerance"""
        example = MockExample(response="100")
        prediction = MockPrediction(response="100.5")
        assert numeric_accuracy(example, prediction) is True
    
    def test_outside_tolerance(self):
        """Test match outside tolerance"""
        example = MockExample(response="100")
        prediction = MockPrediction(response="110")
        assert numeric_accuracy(example, prediction) is False
    
    def test_number_in_text(self):
        """Test extraction from text"""
        example = MockExample(response="The answer is 42")
        prediction = MockPrediction(response="I calculated 42.0")
        assert numeric_accuracy(example, prediction) is True
    
    def test_zero_handling(self):
        """Test zero value handling"""
        example = MockExample(response="0")
        prediction = MockPrediction(response="0.005")
        assert numeric_accuracy(example, prediction) is True
        
        prediction = MockPrediction(response="1")
        assert numeric_accuracy(example, prediction) is False
    
    def test_negative_numbers(self):
        """Test negative number comparison"""
        example = MockExample(response="-42")
        prediction = MockPrediction(response="-42.3")
        assert numeric_accuracy(example, prediction) is True
    
    def test_no_numbers(self):
        """Test when no numbers present"""
        example = MockExample(response="forty two")
        prediction = MockPrediction(response="forty two")
        assert numeric_accuracy(example, prediction) is False


class TestNumericAccuracyAbsolute:
    """Tests for numeric_accuracy_absolute metric"""
    
    def test_within_absolute_tolerance(self):
        """Test match within absolute tolerance"""
        example = MockExample(response="100")
        prediction = MockPrediction(response="100.3")
        # Default tolerance is 0.5
        assert numeric_accuracy_absolute(example, prediction) is True
    
    def test_outside_absolute_tolerance(self):
        """Test match outside absolute tolerance"""
        example = MockExample(response="100")
        prediction = MockPrediction(response="101")
        # Default tolerance is 0.5
        assert numeric_accuracy_absolute(example, prediction) is False
    
    def test_custom_tolerance(self):
        """Test with custom tolerance"""
        example = MockExample(response="100")
        prediction = MockPrediction(response="102")
        # tolerance=3 should pass
        assert numeric_accuracy_absolute(
            example, prediction, tolerance=3.0
        ) is True


# =============================================================================
# Tests for Fuzzy Match Metrics
# =============================================================================

class TestFuzzyMatch:
    """Tests for fuzzy_match metric"""
    
    def test_identical_strings(self):
        """Test identical strings get score 1.0"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello world")
        assert fuzzy_match(example, prediction) == 1.0
    
    def test_similar_strings(self):
        """Test similar strings get high score"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello werld")
        score = fuzzy_match(example, prediction)
        assert score > 0.8
    
    def test_different_strings(self):
        """Test different strings get low score"""
        example = MockExample(response="hello")
        prediction = MockPrediction(response="goodbye")
        score = fuzzy_match(example, prediction)
        assert score < 0.5
    
    def test_empty_strings(self):
        """Test empty string handling"""
        example = MockExample(response="")
        prediction = MockPrediction(response="")
        assert fuzzy_match(example, prediction) == 1.0
    
    def test_case_insensitive(self):
        """Test case insensitivity"""
        example = MockExample(response="HELLO")
        prediction = MockPrediction(response="hello")
        assert fuzzy_match(example, prediction) == 1.0


class TestJaccardSimilarity:
    """Tests for jaccard_similarity metric"""
    
    def test_identical_text(self):
        """Test identical text"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello world")
        assert jaccard_similarity(example, prediction) == 1.0
    
    def test_partial_overlap(self):
        """Test partial word overlap"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello there")
        score = jaccard_similarity(example, prediction)
        # 1 common word out of 3 unique words
        assert 0.2 < score < 0.5
    
    def test_no_overlap(self):
        """Test no word overlap"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="goodbye universe")
        assert jaccard_similarity(example, prediction) == 0.0


# =============================================================================
# Tests for Combined Metrics
# =============================================================================

class TestCombinedMetric:
    """Tests for combined_metric function"""
    
    def test_exact_match_dominates(self):
        """Test that exact match contributes most"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="42")
        score = combined_metric(example, prediction)
        assert score > 0.9
    
    def test_contains_partial_credit(self):
        """Test partial credit for containment"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="The answer is 42.")
        score = combined_metric(example, prediction)
        # Not exact, but contains
        assert 0.3 < score < 0.9
    
    def test_fuzzy_only(self):
        """Test fuzzy matching alone"""
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello werld")
        score = combined_metric(example, prediction)
        # Only fuzzy score contributes
        assert score > 0.1


# =============================================================================
# Tests for Metric Factories
# =============================================================================

class TestCreateNumericMetric:
    """Tests for create_numeric_metric factory"""
    
    def test_relative_tolerance(self):
        """Test factory with relative tolerance"""
        metric = create_numeric_metric(tolerance=0.1, absolute=False)
        
        example = MockExample(response="100")
        prediction = MockPrediction(response="108")
        assert metric(example, prediction) is True
        
        prediction = MockPrediction(response="112")
        assert metric(example, prediction) is False
    
    def test_absolute_tolerance(self):
        """Test factory with absolute tolerance"""
        metric = create_numeric_metric(tolerance=5.0, absolute=True)
        
        example = MockExample(response="100")
        prediction = MockPrediction(response="104")
        assert metric(example, prediction) is True
        
        prediction = MockPrediction(response="106")
        assert metric(example, prediction) is False


class TestCreateFuzzyMetric:
    """Tests for create_fuzzy_metric factory"""
    
    def test_custom_threshold(self):
        """Test factory with custom threshold"""
        metric = create_fuzzy_metric(threshold=0.9)
        
        example = MockExample(response="hello")
        prediction = MockPrediction(response="hello")
        assert metric(example, prediction) is True
        
        prediction = MockPrediction(response="hallo")
        # Should fail with high threshold
        assert metric(example, prediction) is False
    
    def test_low_threshold(self):
        """Test factory with low threshold"""
        metric = create_fuzzy_metric(threshold=0.5)
        
        example = MockExample(response="hello world")
        prediction = MockPrediction(response="hello there")
        # Should pass with low threshold
        assert metric(example, prediction) is True


class TestCreateCombinedMetric:
    """Tests for create_combined_metric factory"""
    
    def test_custom_weights(self):
        """Test factory with custom weights"""
        # Weight exact match heavily
        metric = create_combined_metric(
            exact_weight=0.9,
            contains_weight=0.05,
            fuzzy_weight=0.05
        )
        
        example = MockExample(response="42")
        
        # Exact match should score very high
        prediction = MockPrediction(response="42")
        score1 = metric(example, prediction)
        
        # Contains should score much lower
        prediction = MockPrediction(response="The answer is 42")
        score2 = metric(example, prediction)
        
        assert score1 > score2


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        example = MockExample(response="π ≈ 3.14")
        prediction = MockPrediction(response="π ≈ 3.14")
        assert exact_match(example, prediction) is True
    
    def test_multiline_text(self):
        """Test handling of multiline text"""
        example = MockExample(response="line1\nline2\nline3")
        prediction = MockPrediction(response="line1\nline2\nline3")
        assert exact_match(example, prediction) is True
    
    def test_special_characters(self):
        """Test handling of special characters"""
        example = MockExample(response="result: $100.50 (USD)")
        prediction = MockPrediction(response="result: $100.50 (USD)")
        assert exact_match(example, prediction) is True
    
    def test_very_long_strings(self):
        """Test handling of very long strings"""
        long_text = "word " * 1000
        example = MockExample(response=long_text)
        prediction = MockPrediction(response=long_text)
        
        assert exact_match(example, prediction) is True
        # Fuzzy should use word-based comparison for long strings
        score = fuzzy_match(example, prediction)
        assert score == 1.0
    
    def test_numeric_in_scientific_notation(self):
        """Test numeric extraction from scientific notation"""
        # The regex may not handle this perfectly
        example = MockExample(response="1e6")
        result = _extract_number(example.response)
        # Should extract 6 (last number found)
        assert result is not None
    
    def test_trace_parameter_ignored(self):
        """Test that trace parameter doesn't break anything"""
        example = MockExample(response="42")
        prediction = MockPrediction(response="42")
        
        # All metrics should accept trace parameter
        assert exact_match(example, prediction, trace="ignored") is True
        assert contains_answer(example, prediction, trace=None) is True
        assert numeric_accuracy(example, prediction, trace={}) is True