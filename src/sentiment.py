"""Sentiment analysis for disaster tweets.

Why two analysers?
------------------
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is the primary
tool. It was built specifically for social-media text: it understands
emoticons, slang, ALL-CAPS intensity, degree modifiers ("very"), and - most
importantly for this project - explicit negation handling, so "not safe"
scores differently from "safe". Its lexicon ships inside the ``vaderSentiment``
package, so it needs no download at inference time.

**TextBlob** is reported alongside it because it provides a *subjectivity*
score, which VADER does not. Subjectivity is interesting here: factual
disaster reporting ("7.8 magnitude earthquake strikes Nepal") tends to be more
objective than figurative usage ("this exam destroyed me").

Sentiment is NOT the same task as disaster classification
---------------------------------------------------------
A real disaster report can be written in flat, neutral, factual language and
score ~0.0 compound. A joke about a "trainwreck of a day" can be strongly
negative while being completely non-disaster. Sentiment is therefore used as
*one weak feature among many*, never as a decision rule. The notebook
quantifies exactly how weak the relationship is rather than assuming it.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing import _is_missing

#: Order matters: this is the column order the fitted pipeline expects.
SENTIMENT_FEATURE_NAMES: list[str] = [
    "vader_compound",
    "vader_positive",
    "vader_negative",
    "vader_neutral",
    "textblob_polarity",
    "textblob_subjectivity",
]


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return a cached VADER analyser.

    Constructing the analyser parses its ~7,500-entry lexicon from disk, so it
    is built once and reused. ``lru_cache`` also keeps the object out of the
    pickled model, which keeps the saved artifact small.
    """
    return SentimentIntensityAnalyzer()


def analyse_sentiment(text: object) -> dict[str, float]:
    """Compute VADER and TextBlob sentiment scores for one tweet.

    Sentiment is computed on the **raw** text, not the cleaned text: VADER
    deliberately uses capitalisation ("FIRE" is more intense than "fire"),
    punctuation ("help!!!"), and emoticons, all of which preprocessing removes.

    Returns
    -------
    dict
        ``vader_compound`` in [-1, 1]; ``vader_positive``/``negative``/
        ``neutral`` are proportions summing to 1; ``textblob_polarity`` in
        [-1, 1]; ``textblob_subjectivity`` in [0, 1] where 0 is fully
        objective.

    Examples
    --------
    >>> s = analyse_sentiment("Massive fire destroyed the building!")
    >>> s["vader_compound"] < 0
    True
    """
    if _is_missing(text) or not str(text).strip():
        return {name: 0.0 for name in SENTIMENT_FEATURE_NAMES}

    raw = str(text)
    vader = get_analyzer().polarity_scores(raw)
    blob = TextBlob(raw).sentiment

    return {
        "vader_compound": float(vader["compound"]),
        "vader_positive": float(vader["pos"]),
        "vader_negative": float(vader["neg"]),
        "vader_neutral": float(vader["neu"]),
        "textblob_polarity": float(blob.polarity),
        "textblob_subjectivity": float(blob.subjectivity),
    }


def sentiment_label(compound: float) -> str:
    """Map a VADER compound score to a human-readable label.

    Uses the thresholds documented by the VADER authors (+/-0.05).
    """
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def add_sentiment_features(
    frame: pd.DataFrame, text_column: str = "text"
) -> pd.DataFrame:
    """Return a copy of ``frame`` with sentiment columns appended."""
    if text_column not in frame.columns:
        raise KeyError(
            f"Column {text_column!r} not found. Available: {list(frame.columns)}"
        )

    scores = frame[text_column].apply(analyse_sentiment).apply(pd.Series)
    result = pd.concat([frame.copy(), scores[SENTIMENT_FEATURE_NAMES]], axis=1)
    result["sentiment_label"] = result["vader_compound"].apply(sentiment_label)
    return result


def build_sentiment_frame(texts: list[object]) -> pd.DataFrame:
    """Build a sentiment-only DataFrame from a list of raw tweets."""
    rows = [analyse_sentiment(text) for text in texts]
    return pd.DataFrame(rows, columns=SENTIMENT_FEATURE_NAMES)
