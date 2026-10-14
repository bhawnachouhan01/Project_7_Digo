"""Pipeline factories shared by the training notebook, the tests and the app.

Why a factory module exists
---------------------------
The fitted artifact saved to ``models/selected_model.joblib`` is a complete
scikit-learn :class:`~sklearn.pipeline.Pipeline`: it contains the vectorizer
*and* the classifier. joblib pickles the pipeline **by reference** for any
module-level function it points at, so the callables used as transformers must
live in an importable module - not in the notebook, and not as lambdas.
Putting them here is what makes "fit in the notebook, predict in the app"
work without re-implementing anything.

Because the vectorizer's ``preprocessor`` is :func:`src.preprocessing.clean_text`
itself, there is exactly one cleaning implementation in the project. Training
and inference cannot drift apart - a property asserted directly in
``tests/test_app_logic.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.features import NUMERIC_FEATURE_NAMES, build_feature_frame
from src.preprocessing import clean_text
from src.sentiment import SENTIMENT_FEATURE_NAMES, build_sentiment_frame

#: Column name used for the raw tweet inside combined pipelines.
TEXT_COLUMN = "text"

#: All engineered (non-text) columns, in fixed order.
ENGINEERED_FEATURE_NAMES: list[str] = NUMERIC_FEATURE_NAMES + SENTIMENT_FEATURE_NAMES


class EngineeredFeatureExtractor(BaseEstimator, TransformerMixin):
    """Compute engineered + sentiment features from a raw-text column.

    Implemented as a transformer (rather than precomputed columns) so that the
    features are produced *inside* the pipeline. That means cross-validation
    folds and the Streamlit app both get them computed the same way, from raw
    text, with no chance of a precomputed column going stale.

    These features are deterministic functions of a single tweet - they use no
    corpus-level statistics - so computing them before the train/test split
    would not leak. They are kept inside the pipeline anyway, for consistency
    and so that one ``pipeline.predict([tweet])`` call is all the app needs.
    """

    def fit(self, X: Any, y: Any = None) -> "EngineeredFeatureExtractor":  # noqa: N803
        return self

    def transform(self, X: Any) -> pd.DataFrame:  # noqa: N803
        texts = _as_text_list(X)
        numeric = build_feature_frame(texts)
        sentiment = build_sentiment_frame(texts)
        return pd.concat([numeric, sentiment], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(ENGINEERED_FEATURE_NAMES)


def _as_text_list(X: Any) -> list[object]:  # noqa: N803
    """Coerce whatever the pipeline hands us into a flat list of strings."""
    if isinstance(X, pd.DataFrame):
        return X[TEXT_COLUMN].tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    return list(X)


class TextSelector(BaseEstimator, TransformerMixin):
    """Select the raw-text column from a DataFrame as a 1-D sequence.

    ``ColumnTransformer`` hands a *DataFrame* to each branch, but
    ``TfidfVectorizer`` requires a 1-D iterable of strings. This adapter
    bridges the two.
    """

    def fit(self, X: Any, y: Any = None) -> "TextSelector":  # noqa: N803
        return self

    def transform(self, X: Any) -> list[object]:  # noqa: N803
        return _as_text_list(X)


def make_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 30_000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Build the project's standard TF-IDF vectorizer.

    ``preprocessor=clean_text`` is the critical argument: it wires the
    project's single cleaning implementation directly into the fitted model.

    ``sublinear_tf=True`` applies ``1 + log(tf)``, which damps the effect of a
    word repeated many times in one tweet - appropriate for very short text.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        tokenizer=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        strip_accents=None,
        lowercase=False,  # clean_text already lowercases
    )


def make_count_vectorizer(
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_features: int | None = 30_000,
) -> CountVectorizer:
    """Build a Bag-of-Words (raw count) vectorizer using the same cleaner."""
    return CountVectorizer(
        preprocessor=clean_text,
        tokenizer=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
    )


def make_text_pipeline(classifier: BaseEstimator, vectorizer: Any = None) -> Pipeline:
    """Text-only pipeline: vectorizer -> classifier."""
    return Pipeline(
        [
            ("vectorizer", vectorizer if vectorizer is not None else make_tfidf_vectorizer()),
            ("classifier", classifier),
        ]
    )


def make_combined_pipeline(classifier: BaseEstimator, vectorizer: Any = None) -> Pipeline:
    """Pipeline combining TF-IDF text features with engineered numeric features.

    Uses :class:`~sklearn.compose.ColumnTransformer` so that both branches are
    fitted on training folds only. The numeric branch is standardised because
    raw counts (``char_count`` up to 157) and ratios (0-1) are on wildly
    different scales, which matters for the linear models.

    Expects a DataFrame with a ``text`` column.
    """
    text_branch = Pipeline(
        [
            ("select", TextSelector()),
            ("tfidf", vectorizer if vectorizer is not None else make_tfidf_vectorizer()),
        ]
    )
    numeric_branch = Pipeline(
        [
            ("engineer", EngineeredFeatureExtractor()),
            ("scale", StandardScaler()),
        ]
    )

    combined = ColumnTransformer(
        transformers=[
            ("text", text_branch, [TEXT_COLUMN]),
            ("numeric", numeric_branch, [TEXT_COLUMN]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("features", combined), ("classifier", classifier)])


def default_seed_kwargs(estimator_class: type) -> dict[str, Any]:
    """Return ``{'random_state': SEED}`` if the estimator supports it."""
    if "random_state" in estimator_class().get_params():
        return {"random_state": RANDOM_SEED}
    return {}
