"""Tests for the saved model artifacts.

These run in a **fresh pytest process** with no notebook state, which is the real
proof that the Streamlit app can load and use the model after a ``git clone``.

If the model has not been trained yet, these tests skip with a clear message
rather than failing - the notebook is what produces the artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import METADATA_FILE, MODEL_FILE

pytestmark = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason=f"{MODEL_FILE.name} not found - run notebooks/Project_7_Digo_NLP.ipynb first",
)


@pytest.fixture(scope="module")
def model():
    import joblib

    return joblib.load(MODEL_FILE)


@pytest.fixture(scope="module")
def metadata():
    if not METADATA_FILE.exists():
        pytest.skip("model_metadata.json not found")
    with open(METADATA_FILE, encoding="utf-8") as handle:
        return json.load(handle)


class TestModelLoading:
    def test_model_file_exists(self):
        assert MODEL_FILE.exists()

    def test_model_loads_from_disk(self, model):
        assert model is not None

    def test_model_exposes_predict(self, model):
        assert hasattr(model, "predict")

    def test_model_is_small_enough_for_github(self):
        size_mb = MODEL_FILE.stat().st_size / 1024**2
        assert size_mb < 100, f"{size_mb:.1f} MB exceeds GitHub's 100 MB file limit"

    def test_model_is_already_fitted(self, model):
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(model)
        except NotFittedError:
            pytest.fail("saved model is not fitted - the app would crash on first use")


class TestPredictionOutput:
    SAMPLE_TWEETS = [
        "Massive earthquake hits the city, buildings have collapsed",
        "Just had the best pizza of my life",
        "that exam absolutely destroyed me",
        "Breaking: wildfire forces evacuation of 2000 residents",
        "Can't wait for the weekend",
    ]

    def test_predicts_one_label_per_input(self, model):
        predictions = model.predict(self.SAMPLE_TWEETS)
        assert len(predictions) == len(self.SAMPLE_TWEETS)

    def test_labels_are_only_zero_or_one(self, model):
        predictions = model.predict(self.SAMPLE_TWEETS)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_single_tweet_returns_single_prediction(self, model):
        assert len(model.predict(["A single tweet"])) == 1

    def test_probabilities_are_valid(self, model):
        if not hasattr(model, "predict_proba"):
            pytest.skip("model does not expose predict_proba")
        probabilities = model.predict_proba(self.SAMPLE_TWEETS)
        assert probabilities.shape == (len(self.SAMPLE_TWEETS), 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_probability_agrees_with_predicted_label(self, model):
        if not hasattr(model, "predict_proba"):
            pytest.skip("model does not expose predict_proba")
        predictions = model.predict(self.SAMPLE_TWEETS)
        probabilities = model.predict_proba(self.SAMPLE_TWEETS)[:, 1]
        for prediction, probability in zip(predictions, probabilities):
            assert prediction == int(probability >= 0.5)

    def test_predictions_are_deterministic(self, model):
        first = model.predict(self.SAMPLE_TWEETS)
        second = model.predict(self.SAMPLE_TWEETS)
        assert np.array_equal(first, second)


class TestModelBehaviour:
    """Sanity checks on obvious cases.

    Deliberately limited to unambiguous examples. We do not assert behaviour on
    genuinely ambiguous tweets, because there is no defensible ground truth for them.
    """

    def test_clear_disaster_is_classified_as_disaster(self, model):
        clear_disasters = [
            "Massive earthquake hits the city, buildings collapsed and people are trapped",
            "Breaking: wildfire forces evacuation of 2000 residents in California",
            "Emergency services responding to a major train derailment, several injured",
        ]
        predictions = model.predict(clear_disasters)
        assert predictions.sum() >= 2, (
            "model failed on a majority of unambiguous disaster tweets"
        )

    def test_clear_non_disaster_is_classified_as_non_disaster(self, model):
        clear_non_disasters = [
            "Just had the best pizza of my life, highly recommend this place",
            "Can't wait for the weekend, going to relax and watch movies",
            "Happy birthday to my wonderful sister, hope you have a great day",
        ]
        predictions = model.predict(clear_non_disasters)
        assert predictions.sum() <= 1, (
            "model flagged a majority of clearly harmless tweets as disasters"
        )

    def test_handles_unusual_input_without_crashing(self, model):
        awkward = [
            "a",
            "!!!???",
            "http://t.co/abc123",
            "@user @user2 @user3",
            "#tag " * 20,
            "x" * 400,
            "12345",
            "ALL CAPS SHOUTING TWEET",
        ]
        predictions = model.predict(awkward)
        assert len(predictions) == len(awkward)
        assert set(np.unique(predictions)).issubset({0, 1})


class TestMetadata:
    REQUIRED_KEYS = [
        "model_name", "model_type", "created_at", "dataset",
        "training_config", "vectorizer", "test_metrics", "label_mapping", "versions",
    ]

    def test_metadata_file_exists(self, metadata):
        assert metadata

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_keys_present(self, metadata, key):
        assert key in metadata

    def test_metrics_are_in_valid_range(self, metadata):
        for name, value in metadata["test_metrics"].items():
            if isinstance(value, float):
                assert 0.0 <= value <= 1.0, f"{name}={value} is not a valid score"

    def test_metrics_are_plausible_not_perfect(self, metadata):
        # A perfect score on this dataset would indicate leakage, since human
        # annotators themselves disagreed on some tweets.
        assert metadata["test_metrics"]["accuracy"] < 0.99
        assert metadata["test_metrics"]["accuracy"] > 0.60

    def test_dataset_row_count_matches_the_supplied_file(self, metadata):
        assert metadata["dataset"]["rows_in_source_file"] == 7613

    def test_confusion_matrix_sums_to_test_size(self, metadata):
        matrix = metadata["test_metrics"]["confusion_matrix"]
        total = sum(matrix.values())
        assert total == metadata["test_metrics"]["evaluated_on_rows"]

    def test_label_mapping_is_correct(self, metadata):
        assert metadata["label_mapping"]["1"] == "Disaster-related"
        assert metadata["label_mapping"]["0"] == "Non-disaster"


class TestMissingArtifactHandling:
    """The app must degrade gracefully when artifacts are absent."""

    def test_app_reports_missing_model_without_crashing(self, monkeypatch, tmp_path):
        import app as digo_app

        monkeypatch.setattr(digo_app, "MODEL_FILE", tmp_path / "does_not_exist.joblib")
        digo_app.load_model.clear()

        model, error = digo_app.load_model()
        assert model is None
        assert error and "not found" in error.lower()
        digo_app.load_model.clear()

    def test_app_handles_corrupt_model_file(self, monkeypatch, tmp_path):
        import app as digo_app

        corrupt = tmp_path / "corrupt.joblib"
        corrupt.write_bytes(b"this is not a joblib file")
        monkeypatch.setattr(digo_app, "MODEL_FILE", corrupt)
        digo_app.load_model.clear()

        model, error = digo_app.load_model()
        assert model is None
        assert error
        digo_app.load_model.clear()
