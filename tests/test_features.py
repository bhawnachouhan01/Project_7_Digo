"""Tests for engineered numeric features and sentiment scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    NUMERIC_FEATURE_NAMES,
    add_text_features,
    build_feature_frame,
    extract_text_features,
)
from src.sentiment import (
    SENTIMENT_FEATURE_NAMES,
    analyse_sentiment,
    build_sentiment_frame,
    sentiment_label,
)


class TestFeatureExtraction:
    def test_all_declared_features_are_returned(self):
        features = extract_text_features("A normal tweet about something")
        assert set(features) == set(NUMERIC_FEATURE_NAMES)

    def test_all_values_are_floats(self):
        for value in extract_text_features("Fire! #alert @cnn http://t.co/x").values():
            assert isinstance(value, float)

    @pytest.mark.parametrize("value", [None, "", np.nan, 123])
    def test_missing_input_gives_all_zeros(self, value):
        features = extract_text_features(value)
        assert all(v == 0.0 for v in features.values())

    def test_counts_are_correct(self):
        features = extract_text_features("Fire in CA! #wildfire #alert @CNN http://t.co/x")
        assert features["hashtag_count"] == 2
        assert features["mention_count"] == 1
        assert features["url_count"] == 1
        assert features["exclamation_count"] == 1

    def test_char_and_word_counts(self):
        text = "one two three"
        features = extract_text_features(text)
        assert features["char_count"] == len(text)
        assert features["word_count"] == 3

    def test_uppercase_ratio(self):
        assert extract_text_features("ABCD")["uppercase_ratio"] == 1.0
        assert extract_text_features("abcd")["uppercase_ratio"] == 0.0
        assert extract_text_features("AAbb")["uppercase_ratio"] == pytest.approx(0.5)

    def test_digit_count(self):
        assert extract_text_features("200 dead and 1500 injured")["digit_count"] == 7

    def test_features_measure_raw_text_not_cleaned(self):
        # If features were computed after cleaning, the '#' would be gone and
        # this count would wrongly be zero.
        assert extract_text_features("#wildfire #flood")["hashtag_count"] == 2


class TestFeatureFrames:
    def test_build_feature_frame_shape_and_order(self):
        frame = build_feature_frame(["first tweet", "second tweet", None])
        assert frame.shape == (3, len(NUMERIC_FEATURE_NAMES))
        assert list(frame.columns) == NUMERIC_FEATURE_NAMES

    def test_add_text_features_preserves_original_columns(self):
        original = pd.DataFrame({"text": ["a tweet", "another"], "target": [0, 1]})
        result = add_text_features(original)
        assert "text" in result.columns and "target" in result.columns
        assert len(result) == len(original)

    def test_missing_text_column_raises_clear_error(self):
        with pytest.raises(KeyError, match="not found"):
            add_text_features(pd.DataFrame({"body": ["x"]}))


class TestSentiment:
    def test_all_declared_scores_are_returned(self):
        scores = analyse_sentiment("A tweet")
        assert set(scores) == set(SENTIMENT_FEATURE_NAMES)

    @pytest.mark.parametrize("value", [None, "", "   ", np.nan])
    def test_missing_input_gives_zeros(self, value):
        assert all(v == 0.0 for v in analyse_sentiment(value).values())

    def test_negative_text_scores_negative(self):
        assert analyse_sentiment("Terrible disaster, people died, awful tragedy")["vader_compound"] < 0

    def test_positive_text_scores_positive(self):
        assert analyse_sentiment("What a wonderful, happy, beautiful day!")["vader_compound"] > 0

    def test_scores_stay_in_valid_ranges(self):
        for text in ["Massive fire!", "lovely day", "", "earthquake 7.8"]:
            scores = analyse_sentiment(text)
            assert -1.0 <= scores["vader_compound"] <= 1.0
            assert -1.0 <= scores["textblob_polarity"] <= 1.0
            assert 0.0 <= scores["textblob_subjectivity"] <= 1.0

    def test_vader_handles_negation(self):
        # The whole point of using VADER rather than a naive lexicon.
        positive = analyse_sentiment("I feel safe")["vader_compound"]
        negated = analyse_sentiment("I do not feel safe")["vader_compound"]
        assert negated < positive

    @pytest.mark.parametrize(
        "compound,expected",
        [(0.8, "Positive"), (0.05, "Positive"), (0.0, "Neutral"),
         (-0.04, "Neutral"), (-0.05, "Negative"), (-0.9, "Negative")],
    )
    def test_sentiment_label_thresholds(self, compound, expected):
        assert sentiment_label(compound) == expected

    def test_build_sentiment_frame_shape(self):
        frame = build_sentiment_frame(["happy day", "terrible crash", None])
        assert frame.shape == (3, len(SENTIMENT_FEATURE_NAMES))
        assert list(frame.columns) == SENTIMENT_FEATURE_NAMES


class TestNoLabelLeakage:
    """Features must depend only on the text, never on the label."""

    def test_identical_text_gives_identical_features(self):
        text = "Forest fire near La Ronge"
        assert extract_text_features(text) == extract_text_features(text)
        assert analyse_sentiment(text) == analyse_sentiment(text)

    def test_feature_functions_accept_only_text(self):
        import inspect

        for function in (extract_text_features, analyse_sentiment):
            parameters = set(inspect.signature(function).parameters)
            assert "target" not in parameters and "y" not in parameters
