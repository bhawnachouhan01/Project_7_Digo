"""Text cleaning and tokenization for disaster-tweet classification.

Design notes
------------
**One source of truth.** The functions here are imported by both the training
notebook and ``app.py``. The fitted scikit-learn pipeline references
:func:`clean_text` as the vectorizer's ``preprocessor``, so a tweet typed into
the Streamlit app goes through *byte-for-byte* the same transformation as the
training data. There is no second copy of this logic anywhere in the project.

**No runtime downloads.** NLTK's ``stopwords``/``punkt``/``wordnet`` corpora
require ``nltk.download()``, which is fragile on Streamlit Community Cloud
(no writable cache guarantee, network blips on cold start). So this module
ships its own stopword list and uses a regex tokenizer. ``PorterStemmer`` is
the one NLTK component used, and it is pure Python with no data files.

**Negation is preserved on purpose.** Standard stopword lists delete ``not``,
``no``, ``never``, ``nor`` and ``n't``. For this task that is actively
harmful: "no flooding here" and "flooding here" would collapse to the same
tokens. :data:`NEGATION_WORDS` is therefore subtracted from the stopword list,
and ``n't`` contractions are expanded to the explicit token ``not`` before
tokenization so that TF-IDF bigrams such as ``not safe`` can form.

**Cleaning never looks at the label.** Nothing in this module takes ``target``
as an argument. All rules are derived from the structure of tweet text, not
from the class an example belongs to, so applying them cannot leak the target.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# Vendored stopword list
# ---------------------------------------------------------------------------
# A standard English stopword set. Kept here (rather than downloaded from
# NLTK at runtime) so that inference has zero network dependencies.
_BASE_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can cannot could did do does
    doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more
    most my myself of off on once only or other ought our ours ourselves out
    over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why with would you
    your yours yourself yourselves
    """.split()
)

# Words that flip or qualify meaning. These are deliberately NOT removed.
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "nor",
        "never",
        "none",
        "nothing",
        "nowhere",
        "neither",
        "without",
        "cant",
        "wont",
        "dont",
        "didnt",
        "doesnt",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "shouldnt",
        "wouldnt",
        "couldnt",
    }
)

#: Final stopword set actually used: base list minus anything negating.
STOPWORDS: frozenset[str] = frozenset(_BASE_STOPWORDS - NEGATION_WORDS)

_STEMMER = PorterStemmer()

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@(\w+)")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
# Splits CamelCase inside hashtags: #CaliforniaWildfire -> California Wildfire
CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Thousands separators inside numbers: "13,000" -> "13000". Must run before
# punctuation stripping, otherwise the number splits into "13" and "000".
THOUSANDS_SEP_PATTERN = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Three or more identical *letters* in a row: "sooooo" -> "soo". Restricted to
# letters so that digit runs such as "1000" or "200" are left intact.
REPEAT_CHAR_PATTERN = re.compile(r"([a-z])\1{2,}")
# Keep the decimal point inside numbers ("M5.8 earthquake", "M1.94") - these
# are magnitudes and carry real meaning in this dataset.
NON_ALPHANUM_PATTERN = re.compile(r"(?!(?<=\d)\.(?=\d))[^a-z0-9\s']")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?(?:'[a-z]+)?|[a-z']+")

# Contractions that carry negation. Expanded before punctuation is stripped,
# otherwise "don't" would become the meaningless token "dont"/"don t".
_NEGATION_CONTRACTIONS = [
    (re.compile(r"\bcan['’]?t\b", re.IGNORECASE), "can not"),
    (re.compile(r"\bwon['’]?t\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bshan['’]?t\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bain['’]?t\b", re.IGNORECASE), "is not"),
    (re.compile(r"n['’]t\b", re.IGNORECASE), " not"),
]


def _is_missing(text: object) -> bool:
    """Return True for None, NaN, or anything that is not a string."""
    if text is None:
        return True
    if isinstance(text, float):  # covers numpy.nan / float('nan')
        return text != text
    return not isinstance(text, str)


def normalise_unicode(text: str) -> str:
    """Unescape HTML entities and normalise unicode punctuation.

    Tweets in this dataset contain entities such as ``&amp;``, ``&gt;`` and
    ``&lt;``, plus curly quotes. Left alone, ``&amp;`` becomes the junk token
    ``amp``, which shows up near the top of raw word-frequency counts.
    """
    # Twitter exports are sometimes double-escaped (&amp;amp;), so unescape twice.
    text = html.unescape(html.unescape(text))
    text = unicodedata.normalize("NFKC", text)
    return text.replace("’", "'").replace("‘", "'")


def expand_negation_contractions(text: str) -> str:
    """Rewrite ``don't`` -> ``do not`` so the negation survives tokenization."""
    for pattern, replacement in _NEGATION_CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def split_hashtag_text(match: re.Match[str]) -> str:
    """Convert a hashtag into its readable words.

    ``#CaliforniaWildfire`` -> ``california wildfire``. The hashtag body is
    kept because it is frequently the most informative part of a disaster
    tweet (e.g. ``#wildfires``, ``#earthquake``); only the ``#`` marker is
    dropped. The *count* of hashtags is captured separately in
    :mod:`src.features`, so no information is lost by removing the symbol.
    """
    return " " + CAMEL_SPLIT_PATTERN.sub(" ", match.group(1)) + " "


def clean_text(text: object, stem: bool = False) -> str:
    """Clean a single tweet into a normalised, space-separated token string.

    Steps, in order:

    1. Missing/non-string input becomes an empty string.
    2. HTML entities unescaped, unicode normalised.
    3. URLs replaced with the placeholder token ``urltoken``.
    4. ``@mentions`` replaced with ``usertoken`` (the handle itself is noise,
       but the *fact* of an @-reply is a real signal).
    5. Hashtag bodies kept and CamelCase-split; the ``#`` is dropped.
    6. Negation contractions expanded (``isn't`` -> ``is not``).
    7. Lowercased.
    8. Thousands separators stripped (``13,000`` -> ``13000``).
    9. Runs of 3+ repeated *letters* squashed (``soooo`` -> ``soo``); digit
       runs are left alone.
    10. Remaining punctuation removed; apostrophes and decimal points inside
        numbers are kept.
    11. Tokenized; stopwords removed *except* negation words; 1-character
        tokens dropped unless numeric.
    12. Optionally Porter-stemmed.

    Numbers are deliberately **kept**: magnitudes, death tolls and highway
    numbers ("13,000 people receive evacuation orders", "M5.8 earthquake")
    are meaningful here.

    Parameters
    ----------
    text:
        Raw tweet. ``None``/``NaN``/non-string values are tolerated and
        return ``""``.
    stem:
        Apply Porter stemming to the surviving tokens.

    Returns
    -------
    str
        Space-separated cleaned tokens, possibly empty.

    Examples
    --------
    >>> clean_text("Forest fire near La Ronge Sask. Canada")
    'forest fire near la ronge sask canada'
    >>> clean_text("There is no flooding here")
    'no flooding'
    >>> clean_text(None)
    ''
    """
    if _is_missing(text):
        return ""

    text = normalise_unicode(str(text))
    text = URL_PATTERN.sub(" urltoken ", text)
    text = MENTION_PATTERN.sub(" usertoken ", text)
    text = HASHTAG_PATTERN.sub(split_hashtag_text, text)
    text = expand_negation_contractions(text)

    text = text.lower()
    text = THOUSANDS_SEP_PATTERN.sub("", text)
    text = REPEAT_CHAR_PATTERN.sub(r"\1\1", text)
    text = NON_ALPHANUM_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    tokens = [
        token.strip("'")
        for token in TOKEN_PATTERN.findall(text)
    ]
    tokens = [
        token
        for token in tokens
        if token and (len(token) > 1 or token.isdigit()) and token not in STOPWORDS
    ]

    if stem:
        tokens = [_STEMMER.stem(token) for token in tokens]

    return " ".join(tokens)


def clean_text_stemmed(text: object) -> str:
    """Stemming variant of :func:`clean_text`.

    Exists as a named module-level function (rather than a lambda or
    ``partial``) so that it can be pickled by joblib when used as a
    vectorizer ``preprocessor``.
    """
    return clean_text(text, stem=True)


def tokenize(text: object) -> list[str]:
    """Clean then split a tweet into a list of tokens."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def clean_series(texts: Iterable[object], stem: bool = False) -> list[str]:
    """Apply :func:`clean_text` across an iterable of tweets."""
    return [clean_text(text, stem=stem) for text in texts]


def decode_keyword(keyword: object) -> str:
    """Decode the URL-encoded ``keyword`` column.

    The supplied dataset stores keywords percent-encoded, e.g. ``body%20bags``
    and ``forest%20fire``. Left undecoded they render as single ugly tokens in
    every plot and never match the tweet text.

    >>> decode_keyword("body%20bags")
    'body bags'
    >>> decode_keyword(None)
    ''
    """
    if _is_missing(keyword):
        return ""
    return str(keyword).replace("%20", " ").strip().lower()
