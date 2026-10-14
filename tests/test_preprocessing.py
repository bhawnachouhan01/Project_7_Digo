"""Tests for text cleaning and tokenization.

The behaviours asserted here are the ones the model depends on. If any of these
break, the saved model's input distribution silently changes and its predictions
become unreliable - which is exactly the kind of bug that is hard to notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    NEGATION_WORDS,
    STOPWORDS,
    clean_text,
    clean_text_stemmed,
    decode_keyword,
    tokenize,
)


class TestMissingAndEmptyInput:
    """Cleaning must never raise on bad input - it is fed raw user data."""

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "\n\t  ", float("nan"), np.nan, 123, 45.6, [], {}],
    )
    def test_returns_string_without_raising(self, value):
        result = clean_text(value)
        assert isinstance(result, str)

    @pytest.mark.parametrize("value", [None, "", "   ", np.nan])
    def test_empty_input_gives_empty_output(self, value):
        assert clean_text(value) == ""

    def test_tokenize_of_empty_is_empty_list(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestNegationPreserved:
    """The single most important preprocessing guarantee for this task."""

    @pytest.mark.parametrize("word", ["no", "not", "never", "nor", "none", "without"])
    def test_negation_words_are_not_stopwords(self, word):
        assert word not in STOPWORDS, f"{word!r} must survive stopword removal"

    def test_no_flooding_keeps_the_no(self):
        assert "no" in clean_text("There is no flooding here").split()

    def test_negation_changes_the_output(self):
        with_negation = clean_text("There is no flooding here")
        without_negation = clean_text("There is flooding here")
        assert with_negation != without_negation

    @pytest.mark.parametrize(
        "text",
        ["I don't feel safe", "The building isn't stable", "It can't be real",
         "They won't evacuate", "We haven't been warned"],
    )
    def test_contractions_expand_to_not(self, text):
        assert "not" in clean_text(text).split(), f"negation lost in {text!r}"

    def test_negation_words_are_declared_consistently(self):
        assert NEGATION_WORDS.isdisjoint(STOPWORDS)


class TestUrlsMentionsHashtags:
    def test_url_becomes_placeholder(self):
        result = clean_text("Check this http://t.co/abc123 out")
        assert "urltoken" in result
        assert "http" not in result and "t.co" not in result

    def test_https_and_www_both_handled(self):
        assert "urltoken" in clean_text("See https://example.com/page")
        assert "urltoken" in clean_text("See www.example.com/page")

    def test_mention_becomes_placeholder(self):
        result = clean_text("Hey @CNN look at this")
        assert "usertoken" in result
        assert "cnn" not in result

    def test_hashtag_word_is_kept(self):
        result = clean_text("Evacuations now #wildfire")
        assert "wildfire" in result.split()
        assert "#" not in result

    def test_camelcase_hashtag_is_split(self):
        result = clean_text("#CaliforniaWildfire spreading")
        assert "california" in result.split()
        assert "wildfire" in result.split()


class TestNumbers:
    """Numbers carry real meaning here (death tolls, magnitudes)."""

    def test_plain_numbers_survive(self):
        assert "200" in clean_text("200 people were evacuated").split()

    def test_thousands_separator_is_joined_not_split(self):
        tokens = clean_text("13,000 people receive evacuation orders").split()
        assert "13000" in tokens
        assert "13" not in tokens and "000" not in tokens

    def test_decimal_magnitude_survives(self):
        assert "5.8" in clean_text("Magnitude 5.8 earthquake hit").split()

    def test_digit_runs_are_not_squashed_as_repeats(self):
        # The repeated-character rule must not corrupt "1000" into "100".
        assert "1000" in clean_text("A 1000 year flood").split()


class TestNormalisation:
    def test_html_entities_are_unescaped(self):
        result = clean_text("Fire &amp; smoke everywhere")
        assert "amp" not in result.split()

    def test_double_escaped_entities_are_handled(self):
        assert "amp" not in clean_text("Fire &amp;amp; smoke").split()

    def test_output_is_lowercase(self):
        result = clean_text("MASSIVE FIRE Destroyed Everything")
        assert result == result.lower()

    def test_repeated_letters_are_squashed(self):
        assert "soo" in clean_text("that was sooooooo bad").split()

    def test_whitespace_is_collapsed(self):
        assert "  " not in clean_text("fire    and     smoke")

    def test_punctuation_only_input_is_emptied(self):
        assert clean_text("!!!???...") == ""


class TestStopwords:
    def test_common_stopwords_are_removed(self):
        tokens = clean_text("the fire is in the building").split()
        assert "the" not in tokens and "is" not in tokens

    def test_content_words_survive(self):
        tokens = clean_text("the fire is in the building").split()
        assert "fire" in tokens and "building" in tokens

    def test_single_characters_are_dropped_but_digits_kept(self):
        tokens = clean_text("a b c 5 fire").split()
        assert "a" not in tokens and "b" not in tokens
        assert "fire" in tokens


class TestStemming:
    def test_stemming_is_off_by_default(self):
        assert "flooding" in clean_text("massive flooding reported").split()

    def test_stemming_can_be_enabled(self):
        assert "flood" in clean_text("massive flooding reported", stem=True).split()

    def test_stemmed_helper_matches_the_flag(self):
        text = "buildings were destroyed by flooding"
        assert clean_text_stemmed(text) == clean_text(text, stem=True)


class TestDeterminism:
    """Train/inference consistency depends on this being a pure function."""

    def test_same_input_gives_same_output(self):
        text = "Forest fire near La Ronge Sask. Canada"
        assert clean_text(text) == clean_text(text) == clean_text(text)

    def test_tokenize_matches_clean_text_split(self):
        text = "Breaking: wildfire forces evacuation #CAfire http://t.co/x"
        assert tokenize(text) == clean_text(text).split()


class TestDecodeKeyword:
    def test_url_encoded_space_is_decoded(self):
        assert decode_keyword("body%20bags") == "body bags"

    def test_plain_keyword_passes_through(self):
        assert decode_keyword("earthquake") == "earthquake"

    @pytest.mark.parametrize("value", [None, "", np.nan, float("nan")])
    def test_missing_keyword_gives_empty_string(self, value):
        assert decode_keyword(value) == ""

    def test_result_is_lowercased_and_stripped(self):
        assert decode_keyword("  FOREST%20FIRE  ") == "forest fire"


class TestRealWorldExamples:
    """Cases drawn from the actual dataset."""

    @pytest.mark.parametrize(
        "tweet",
        [
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all",
            "Forest fire near La Ronge Sask. Canada",
            "13,000 people receive #wildfires evacuation orders in California ",
            "M1.94 [01:04 UTC]?5km S of Volcano Hawaii. http://t.co/zDtoyd8EbJ",
            "#RockyFire Update => California Hwy. 20 closed in both directions",
        ],
    )
    def test_real_tweets_produce_non_empty_output(self, tweet):
        result = clean_text(tweet)
        assert result and len(result.split()) >= 2
