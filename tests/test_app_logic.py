"""Tests for the Streamlit app's logic, independent of the UI.

``validate_input`` and ``predict_tweet`` are deliberately written without any
Streamlit calls so they can be imported and tested directly here.

The most important test in this file is
:class:`TestTrainingInferenceConsistency`, which asserts that the model's own
vectorizer uses the project's single ``clean_text`` implementation. That is what
makes "trained in the notebook, served by the app" safe.
"""

from __future__ import annotations

import pytest

from src.config import MAX_INPUT_CHARS, MIN_INPUT_CHARS, MODEL_FILE
from src.preprocessing import clean_text

import app as digo_app


class TestInputValidation:
    @pytest.mark.parametrize(
        "text",
        [
            "Forest fire near La Ronge Sask. Canada",
            "Just had great pizza",
            "Breaking: wildfire forces evacuation of 2000 residents",
            "abc",
        ],
    )
    def test_valid_input_is_accepted(self, text):
        is_valid, message = digo_app.validate_input(text)
        assert is_valid, f"rejected valid input {text!r}: {message}"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
    def test_empty_input_is_rejected(self, text):
        is_valid, message = digo_app.validate_input(text)
        assert not is_valid and message

    def test_too_short_input_is_rejected(self):
        is_valid, _ = digo_app.validate_input("a")
        assert not is_valid

    def test_too_long_input_is_rejected(self):
        is_valid, message = digo_app.validate_input("word " * 500)
        assert not is_valid
        assert str(MAX_INPUT_CHARS) in message.replace(",", "")

    def test_boundary_lengths(self):
        assert digo_app.validate_input("x" * MIN_INPUT_CHARS)[0] is True
        assert digo_app.validate_input("x" * (MAX_INPUT_CHARS + 1))[0] is False

    def test_content_free_input_is_rejected(self):
        # Nothing survives cleaning, so there is nothing to classify.
        is_valid, message = digo_app.validate_input("!!! ??? ...")
        assert not is_valid and "no content" in message.lower()

    def test_every_rejection_explains_itself(self):
        for text in ["", "   ", "a", "!!!", "x" * 9999]:
            is_valid, message = digo_app.validate_input(text)
            assert not is_valid
            assert len(message) > 10, "rejection message should be helpful"


@pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="model not trained yet - run the notebook first",
)
class TestPredictTweet:
    @pytest.fixture(scope="class")
    def model(self):
        import joblib

        return joblib.load(MODEL_FILE)

    def test_result_contains_expected_keys(self, model):
        result = digo_app.predict_tweet(model, "Massive fire downtown")
        for key in ("label_index", "label", "probability", "cleaned_text",
                    "sentiment", "features"):
            assert key in result

    def test_label_matches_label_index(self, model):
        result = digo_app.predict_tweet(model, "Earthquake destroys buildings")
        expected = {0: "Non-disaster", 1: "Disaster-related"}[result["label_index"]]
        assert result["label"] == expected

    def test_label_index_is_binary(self, model):
        for text in ["a fire", "nice day", "flooding everywhere"]:
            assert digo_app.predict_tweet(model, text)["label_index"] in (0, 1)

    def test_probability_is_valid_or_none(self, model):
        probability = digo_app.predict_tweet(model, "Flooding in the city")["probability"]
        assert probability is None or 0.0 <= probability <= 1.0

    def test_cleaned_text_matches_project_cleaner(self, model):
        text = "Check this out http://t.co/abc @CNN #wildfire"
        result = digo_app.predict_tweet(model, text)
        assert result["cleaned_text"] == clean_text(text)

    def test_sentiment_is_computed_not_faked(self, model):
        negative = digo_app.predict_tweet(model, "Horrible tragedy, many people died")
        positive = digo_app.predict_tweet(model, "What a wonderful happy beautiful day")
        assert negative["sentiment"]["vader_compound"] < positive["sentiment"]["vader_compound"]

    def test_prediction_is_not_hardcoded(self, model):
        # Different inputs must be able to produce different outputs.
        labels = {
            digo_app.predict_tweet(model, text)["label_index"]
            for text in [
                "Massive earthquake hits the city, buildings collapsed",
                "Just had the best pizza of my life, highly recommend",
                "Breaking: wildfire forces evacuation of 2000 residents",
                "Happy birthday to my wonderful sister",
            ]
        }
        assert len(labels) == 2, "model returned the same label for everything"


@pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="model not trained yet - run the notebook first",
)
class TestTrainingInferenceConsistency:
    """The guarantee that training and serving cannot drift apart."""

    @pytest.fixture(scope="class")
    def model(self):
        import joblib

        return joblib.load(MODEL_FILE)

    def test_pipeline_vectorizer_uses_project_cleaner(self, model):
        vectorizer = model.named_steps["vectorizer"]
        assert vectorizer.preprocessor is clean_text, (
            "the saved model does not use src.preprocessing.clean_text - "
            "training and inference preprocessing could diverge"
        )

    def test_vectorizer_does_not_double_lowercase(self, model):
        # clean_text already lowercases; the vectorizer must not re-do it.
        assert model.named_steps["vectorizer"].lowercase is False

    def test_identical_text_gives_identical_prediction(self, model):
        text = "Forest fire near La Ronge Sask. Canada"
        first = digo_app.predict_tweet(model, text)
        second = digo_app.predict_tweet(model, text)
        assert first["label_index"] == second["label_index"]
        assert first["probability"] == second["probability"]

    def test_app_and_pipeline_agree(self, model):
        text = "Flooding reported across the city centre"
        assert digo_app.predict_tweet(model, text)["label_index"] == int(model.predict([text])[0])


class TestMetadataLoading:
    def test_metadata_loads_as_dict(self):
        digo_app.load_metadata.clear()
        assert isinstance(digo_app.load_metadata(), dict)

    def test_missing_metadata_returns_empty_dict(self, monkeypatch, tmp_path):
        monkeypatch.setattr(digo_app, "METADATA_FILE", tmp_path / "absent.json")
        digo_app.load_metadata.clear()
        assert digo_app.load_metadata() == {}
        digo_app.load_metadata.clear()

    def test_corrupt_metadata_returns_empty_dict(self, monkeypatch, tmp_path):
        corrupt = tmp_path / "bad.json"
        corrupt.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(digo_app, "METADATA_FILE", corrupt)
        digo_app.load_metadata.clear()
        assert digo_app.load_metadata() == {}
        digo_app.load_metadata.clear()


class TestExampleTweets:
    def test_examples_are_defined(self):
        assert len(digo_app.EXAMPLE_TWEETS) >= 3

    def test_every_example_passes_validation(self):
        for name, text in digo_app.EXAMPLE_TWEETS.items():
            is_valid, message = digo_app.validate_input(text)
            assert is_valid, f"example {name!r} fails validation: {message}"
