"""Engineered numerical features derived from raw tweet text.

Preprocessing vs. feature engineering
-------------------------------------
These are two *different* jobs and the project keeps them apart:

* :mod:`src.preprocessing` **transforms** text into a normalised form that a
  vectorizer can consume. It throws information away (case, punctuation, URLs).
* This module **measures** the raw tweet and produces new numeric columns.
  It must therefore run on the **original** text, before cleaning - counting
  hashtags after the ``#`` has been stripped would always return zero.

Every feature here is computed from the tweet string alone. None of them look
at ``target``, so they cannot leak the label into the model.
"""

from __future__ import annotations

import re
import string

import pandas as pd

from src.preprocessing import URL_PATTERN, _is_missing, clean_text

HASHTAG_COUNT_PATTERN = re.compile(r"#\w+")
MENTION_COUNT_PATTERN = re.compile(r"@\w+")
WORD_SPLIT_PATTERN = re.compile(r"\S+")

_PUNCTUATION = set(string.punctuation)

#: Order matters: this is the column order the fitted pipeline expects.
NUMERIC_FEATURE_NAMES: list[str] = [
    "char_count",
    "word_count",
    "avg_word_length",
    "hashtag_count",
    "mention_count",
    "url_count",
    "digit_count",
    "exclamation_count",
    "question_count",
    "punctuation_count",
    "uppercase_ratio",
    "stopword_ratio",
    "unique_word_ratio",
]


def extract_text_features(text: object) -> dict[str, float]:
    """Compute all engineered numeric features for one raw tweet.

    Parameters
    ----------
    text:
        The **raw**, uncleaned tweet. ``None``/``NaN`` yields all-zero
        features rather than raising, so the function is safe to map over a
        column containing missing values.

    Returns
    -------
    dict
        Keys match :data:`NUMERIC_FEATURE_NAMES`.

    Examples
    --------
    >>> f = extract_text_features("Fire in CA! #wildfire @CNN http://t.co/x")
    >>> f["hashtag_count"], f["mention_count"], f["url_count"]
    (1.0, 1.0, 1.0)
    """
    if _is_missing(text):
        return {name: 0.0 for name in NUMERIC_FEATURE_NAMES}

    raw = str(text)
    words = WORD_SPLIT_PATTERN.findall(raw)
    word_count = len(words)
    char_count = len(raw)

    letters = [char for char in raw if char.isalpha()]
    uppercase_ratio = (
        sum(1 for char in letters if char.isupper()) / len(letters) if letters else 0.0
    )

    avg_word_length = (
        sum(len(word) for word in words) / word_count if word_count else 0.0
    )

    # Ratios computed against the cleaned token stream, which tells us how much
    # signal survives preprocessing. A tweet that is almost entirely stopwords
    # and mentions loses nearly everything.
    cleaned_tokens = clean_text(raw).split()
    stopword_ratio = (
        1.0 - (len(cleaned_tokens) / word_count) if word_count else 0.0
    )
    unique_word_ratio = (
        len(set(cleaned_tokens)) / len(cleaned_tokens) if cleaned_tokens else 0.0
    )

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "avg_word_length": float(avg_word_length),
        "hashtag_count": float(len(HASHTAG_COUNT_PATTERN.findall(raw))),
        "mention_count": float(len(MENTION_COUNT_PATTERN.findall(raw))),
        "url_count": float(len(URL_PATTERN.findall(raw))),
        "digit_count": float(sum(1 for char in raw if char.isdigit())),
        "exclamation_count": float(raw.count("!")),
        "question_count": float(raw.count("?")),
        "punctuation_count": float(sum(1 for char in raw if char in _PUNCTUATION)),
        "uppercase_ratio": float(uppercase_ratio),
        "stopword_ratio": float(max(0.0, stopword_ratio)),
        "unique_word_ratio": float(unique_word_ratio),
    }


def add_text_features(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy of ``frame`` with the engineered feature columns appended.

    Parameters
    ----------
    frame:
        DataFrame containing the raw tweets.
    text_column:
        Name of the column holding raw tweet text.
    """
    if text_column not in frame.columns:
        raise KeyError(
            f"Column {text_column!r} not found. Available: {list(frame.columns)}"
        )

    features = frame[text_column].apply(extract_text_features).apply(pd.Series)
    return pd.concat([frame.copy(), features[NUMERIC_FEATURE_NAMES]], axis=1)


def build_feature_frame(texts: list[object]) -> pd.DataFrame:
    """Build a feature-only DataFrame from a list of raw tweets.

    Used by the inference path in ``app.py`` and inside the fitted pipeline,
    so that a single tweet is featurised exactly as the training rows were.
    """
    rows = [extract_text_features(text) for text in texts]
    return pd.DataFrame(rows, columns=NUMERIC_FEATURE_NAMES)
