"""Digo - Disaster Tweet Classifier (Streamlit application).

Loads the model trained in ``notebooks/Project_7_Digo_NLP.ipynb`` and classifies a
tweet as disaster-related or not.

Design points
-------------
* The model is **loaded, never trained** here. Training happens in the notebook.
  ``@st.cache_resource`` keeps it in memory across reruns.
* Preprocessing is imported from ``src.preprocessing`` - the same code the model was
  fitted with - so a tweet typed here is processed exactly as the training data was.
* Every performance figure displayed comes from ``models/model_metadata.json``, which
  the notebook writes from real evaluation results. Nothing is hardcoded.

Run locally with::

    streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st

from src.config import (
    LABEL_NAMES,
    MAX_INPUT_CHARS,
    METADATA_FILE,
    MIN_INPUT_CHARS,
    MODEL_FILE,
)
from src.features import extract_text_features
from src.logger import log_error, log_prediction
from src.preprocessing import clean_text
from src.sentiment import analyse_sentiment, sentiment_label

# ---------------------------------------------------------------------------
# Student / project profile - edit these to update the About panels
# ---------------------------------------------------------------------------
STUDENT = {
    "name": "Bhawna Chouhan",
    "qualification": "Computer Science Engineering Graduate",
    "role": "Aspiring Data Scientist & AI Professional",
    "project_number": "Project 7",
    "programme": "Data Science & AI Internship",
    "interests": (
        "Data Analytics, Machine Learning, Artificial Intelligence, "
        "Natural Language Processing and Predictive Modelling"
    ),
    # Fill these in with your own links before sharing the app.
    "github": "",
    "linkedin": "",
    "email": "",
}

APP_TITLE = "Digo — Disaster Tweet Classifier"
APP_TAGLINE = "NLP-Powered Disaster Tweet Classification System"

st.set_page_config(
    page_title="Digo - Disaster Tweet Classifier",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Example tweets offered in the UI
# ---------------------------------------------------------------------------
EXAMPLE_TWEETS: dict[str, str] = {
    "Clear disaster": (
        "Breaking: wildfire forces evacuation of 2000 residents in northern California"
    ),
    "Figurative use": "that exam absolutely destroyed me, I need to sleep for a week",
    "Negation": "There is no flooding in our area, everything is fine",
    "Ambiguous": "The forest got destroyed",
    "Everyday chat": "Just had the best pizza of my life, highly recommend this place",
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- page rhythm ---- */
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}

/* ---- hero ---- */
.digo-hero {
    background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 45%, #ea580c 100%);
    border-radius: 16px;
    padding: 2.1rem 2.3rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(185, 28, 28, 0.25);
}
.digo-hero h1 {
    color: #ffffff; font-size: 2.45rem; font-weight: 800;
    margin: 0 0 .35rem 0; letter-spacing: -0.5px; line-height: 1.15;
}
.digo-hero p.tagline {
    color: rgba(255,255,255,.93); font-size: 1.12rem;
    font-weight: 600; margin: 0 0 .9rem 0;
}
.digo-hero p.byline {
    color: rgba(255,255,255,.82); font-size: .93rem; margin: 0;
}
.digo-chip {
    display: inline-block; background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.28); color: #fff;
    padding: .2rem .7rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; margin-right: .45rem;
}

/* ---- result banners ---- */
.result-card {
    border-radius: 14px; padding: 1.5rem 1.7rem; margin: .4rem 0 1rem 0;
    border-left: 7px solid;
}
.result-disaster {background: rgba(220, 38, 38, .10); border-left-color: #dc2626;}
.result-safe     {background: rgba(22, 163, 74, .10); border-left-color: #16a34a;}
.result-card .verdict {font-size: 1.65rem; font-weight: 800; margin: 0 0 .3rem 0;}
.result-disaster .verdict {color: #dc2626;}
.result-safe .verdict {color: #16a34a;}
.result-card .explain {font-size: .97rem; opacity: .85; margin: 0;}

/* ---- section headings ---- */
.section-title {
    font-size: 1.22rem; font-weight: 700;
    margin: 1.6rem 0 .5rem 0; padding-bottom: .35rem;
    border-bottom: 2px solid rgba(185, 28, 28, .35);
}

/* ---- footer ---- */
.digo-footer {
    text-align: center; opacity: .62; font-size: .85rem;
    margin-top: 2.6rem; padding-top: 1.1rem;
    border-top: 1px solid rgba(128,128,128,.25);
}
/* tighten Streamlit's default metric label */
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
</style>
"""


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the fitted pipeline from disk.

    Returns ``(model, error_message)``. A missing or unreadable artifact returns
    ``(None, message)`` so the UI can show a helpful instruction instead of a
    traceback.
    """
    if not MODEL_FILE.exists():
        return None, (
            f"Model file not found at `{MODEL_FILE.name}`. "
            "Run the notebook `notebooks/Project_7_Digo_NLP.ipynb` to train and save it."
        )
    try:
        import joblib

        model = joblib.load(MODEL_FILE)
    except Exception as error:  # corrupt file, version mismatch, etc.
        log_error("Failed to load model", error)
        return None, (
            "The saved model could not be loaded. It may have been created with a "
            "different scikit-learn version. Re-run the notebook to regenerate it."
        )

    if not hasattr(model, "predict"):
        return None, "The loaded artifact is not a usable model."
    return model, None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load evaluation metadata. Returns an empty dict if unavailable."""
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        log_error("Failed to load metadata", error)
        return {}


# ---------------------------------------------------------------------------
# Prediction logic (kept free of Streamlit calls so tests can import it)
# ---------------------------------------------------------------------------
def validate_input(text: str) -> tuple[bool, str]:
    """Check user input before prediction.

    Returns ``(is_valid, message)``.
    """
    if text is None:
        return False, "Please enter a tweet."

    stripped = text.strip()
    if not stripped:
        return False, "Please enter a tweet before clicking Classify."
    if len(stripped) < MIN_INPUT_CHARS:
        return False, f"That is too short - please enter at least {MIN_INPUT_CHARS} characters."
    if len(stripped) > MAX_INPUT_CHARS:
        return False, (
            f"That is {len(stripped):,} characters. Please keep it under "
            f"{MAX_INPUT_CHARS:,} - this model was trained on short tweets."
        )
    if not clean_text(stripped):
        return False, (
            "After removing punctuation, links and common words there is no content "
            "left to analyse. Please enter a tweet with some words in it."
        )
    return True, ""


def predict_tweet(model, text: str) -> dict:
    """Classify one tweet.

    Returns a dict with the label, optional probability, the cleaned text and the
    sentiment scores. ``probability`` is ``None`` when the model genuinely cannot
    produce one - the UI then omits the confidence display rather than inventing it.
    """
    label_index = int(model.predict([text])[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([text])[0][1])

    return {
        "label_index": label_index,
        "label": LABEL_NAMES[label_index],
        "probability": probability,
        "cleaned_text": clean_text(text),
        "sentiment": analyse_sentiment(text),
        "features": extract_text_features(text),
    }


# ---------------------------------------------------------------------------
# Button callbacks
# ---------------------------------------------------------------------------
def _use_example(text: str) -> None:
    """Button callback: load an example into the text area.

    Session state for a widget key may only be modified *before* that widget is
    instantiated, or from inside an ``on_click`` callback (which Streamlit runs
    before the script reruns). Using callbacks here is what keeps the Clear
    button from raising ``StreamlitAPIException``.
    """
    st.session_state["tweet_input"] = text


def _clear_input() -> None:
    """Button callback: empty the text area."""
    st.session_state["tweet_input"] = ""


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_hero() -> None:
    """Branded header block."""
    st.markdown(
        f"""
        <div class="digo-hero">
            <h1>🌀 {STUDENT['name'].split()[0]}'s Disaster Tweet Classifier</h1>
            <p class="tagline">{APP_TAGLINE}</p>
            <p class="byline">
                <span class="digo-chip">{STUDENT['project_number']}</span>
                <span class="digo-chip">{STUDENT['programme']}</span>
                <span class="digo-chip">NLP · Machine Learning</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_panels(metadata: dict) -> None:
    """The About Me / About the Project / Project Overview expanders."""
    dataset = metadata.get("dataset", {})
    metrics = metadata.get("test_metrics", {})
    rows = dataset.get("rows_in_source_file", 0)
    modelled = dataset.get("rows_after_deduplication", 0)

    with st.expander("👩‍🎓  About Me", expanded=False):
        st.markdown(
            f"""
**{STUDENT['name']}** is a {STUDENT['qualification']} and {STUDENT['role']}
with an interest in {STUDENT['interests']}.

This application is part of my **{STUDENT['programme']}** project work, where I apply
data auditing, text preprocessing, exploratory data analysis, feature engineering,
sentiment analysis, and both classical and transformer-based machine learning to solve
a real-world Natural Language Processing problem.
"""
        )
        links = []
        if STUDENT["github"]:
            links.append(f"[GitHub]({STUDENT['github']})")
        if STUDENT["linkedin"]:
            links.append(f"[LinkedIn]({STUDENT['linkedin']})")
        if STUDENT["email"]:
            links.append(f"[Email](mailto:{STUDENT['email']})")
        if links:
            st.markdown("**Connect:** " + "  ·  ".join(links))

    with st.expander("📌  About the Project", expanded=False):
        st.markdown(
            """
**Digo — Disaster Tweet Classification** is an end-to-end Natural Language Processing
project that predicts whether a tweet refers to a **real disaster** or simply uses
disaster-related language figuratively.

During an emergency, social media is often faster than official channels, so relief
agencies and newsrooms would like to monitor it automatically. The difficulty is that
the vocabulary of disaster is also the vocabulary of everyday exaggeration:
"""
        )
        st.markdown(
            """
| Tweet | Contains | Real disaster? |
|---|---|---|
| "Forest fire near La Ronge Sask. Canada" | *fire* | ✅ **Yes** |
| "that exam absolutely destroyed me" | *destroyed* | ❌ No |
| "My new song is FIRE 🔥" | *fire* | ❌ No |
| "13,000 people receive wildfires evacuation orders" | *wildfires* | ✅ **Yes** |
"""
        )
        st.markdown(
            """
A system that simply searches for the word *fire* flags all four. The real task is
**contextual understanding**, which is what makes this a genuine NLP problem rather
than a keyword lookup.
"""
        )

    with st.expander("🎯  Project Overview", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(
                f"""
**Objective**

Classify each tweet as `1` = disaster-related or `0` = non-disaster, handling
figurative, metaphorical and sarcastic uses of disaster words.

**Dataset**

{rows:,} hand-labelled tweets supplied with the project brief
({modelled:,} used after removing duplicates and contradictory labels).
Class balance is 57% non-disaster / 43% disaster.

**Workflow**

Data audit → EDA → text preprocessing → keyword analysis → feature engineering →
sentiment analysis → vectorization → leakage-safe split → model comparison →
DistilBERT → evaluation → deployment.
"""
            )
        with right:
            st.markdown(
                f"""
**Approach**

- **Preprocessing** that preserves meaning — negation words such as *no* and *not*
  are deliberately kept, because *"no flooding"* means the opposite of *"flooding"*.
- **TF-IDF** vectorization over cleaned text.
- **13 engineered features** plus VADER & TextBlob sentiment, evaluated for value.
- **14 models compared** on an untouched test set, including a genuinely fine-tuned
  DistilBERT transformer.

**Result**

{f"Final model reaches **{metrics['accuracy']:.1%} accuracy** and **{metrics['f1']:.3f} F1** on {metrics['evaluated_on_rows']:,} held-out tweets never seen during training." if metrics else "Run the notebook to generate evaluation metrics."}
"""
            )


def render_sidebar(metadata: dict) -> None:
    with st.sidebar:
        st.markdown(f"### 🌀 {APP_TITLE.split('—')[0].strip()}")
        st.caption(APP_TAGLINE)
        st.divider()

        st.markdown("#### 🧠 Model")
        if metadata:
            st.markdown(f"**{metadata.get('model_name', 'n/a')}**")
            st.caption(metadata.get("model_type", ""))
        else:
            st.caption("Model metadata unavailable.")

        st.markdown("#### 📊 Measured performance")
        metrics = metadata.get("test_metrics", {})
        if metrics:
            left, right = st.columns(2)
            left.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            right.metric("F1 (disaster)", f"{metrics['f1']:.3f}")
            left.metric("Precision", f"{metrics['precision']:.1%}")
            right.metric("Recall", f"{metrics['recall']:.1%}")
            rows = metrics.get("evaluated_on_rows")
            st.caption(
                f"Measured on {rows:,} held-out tweets never seen during training."
                if rows else "Measured on a held-out test set."
            )
        else:
            st.caption("Run the notebook to generate evaluation metrics.")

        st.markdown("#### 🗂️ Dataset")
        dataset = metadata.get("dataset", {})
        if dataset:
            st.markdown(
                f"**{dataset.get('rows_in_source_file', 0):,}** labelled tweets  \n"
                f"**{dataset.get('rows_after_deduplication', 0):,}** used after cleaning"
            )
        st.caption(
            "Source: the dataset supplied with the project brief "
            "(Kaggle *NLP with Disaster Tweets* training split)."
        )

        st.divider()
        st.markdown("#### 👩‍🎓 Author")
        st.markdown(f"**{STUDENT['name']}**")
        st.caption(f"{STUDENT['project_number']} · {STUDENT['programme']}")

        st.divider()
        st.warning(
            "This is a student project. Predictions are **not** a substitute for "
            "verified emergency information.",
            icon="⚠️",
        )


def render_result(result: dict) -> None:
    """Render the prediction outcome."""
    is_disaster = result["label_index"] == 1
    probability = result["probability"]

    icon = "🚨" if is_disaster else "✅"
    css = "result-disaster" if is_disaster else "result-safe"
    explain = (
        "The model reads this as referring to a genuine disaster or emergency."
        if is_disaster
        else "The model reads this as everyday, figurative or unrelated language."
    )
    st.markdown(
        f"""
        <div class="result-card {css}">
            <p class="verdict">{icon} {result['label']}</p>
            <p class="explain">{explain}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability is not None:
        confidence = probability if is_disaster else 1 - probability
        st.progress(
            min(max(confidence, 0.0), 1.0),
            text=f"Model confidence: {confidence:.1%}",
        )

        if confidence < 0.65:
            st.warning(
                "**The model is not very confident here.** This tweet sits close to the "
                "decision boundary - often a sign of genuinely ambiguous wording. "
                "Treat this result with caution.",
                icon="⚠️",
            )

        st.caption(
            f"Estimated probability that this tweet is disaster-related: "
            f"**{probability:.1%}**. This is a calibrated probability from the model, "
            "not a measure of whether the event is real."
        )

    detail, sentiment_tab = st.tabs(["🔍 How the model read this", "💬 Sentiment (separate)"])

    with detail:
        st.markdown("**Text after preprocessing** — what the model actually sees:")
        st.code(result["cleaned_text"] or "(nothing left after cleaning)", language=None)
        st.caption(
            "URLs become `urltoken`, @mentions become `usertoken`, hashtag words are "
            "kept, and common words are removed - but negation words like *no* and "
            "*not* are deliberately preserved, because they change meaning."
        )

        features = result["features"]
        st.markdown("**Measured properties of the original tweet:**")
        columns = st.columns(5)
        columns[0].metric("Words", int(features["word_count"]))
        columns[1].metric("Characters", int(features["char_count"]))
        columns[2].metric("Hashtags", int(features["hashtag_count"]))
        columns[3].metric("Mentions", int(features["mention_count"]))
        columns[4].metric("Links", int(features["url_count"]))

    with sentiment_tab:
        sentiment = result["sentiment"]
        compound = sentiment["vader_compound"]
        columns = st.columns(3)
        columns[0].metric("Tone", sentiment_label(compound))
        columns[1].metric("VADER compound", f"{compound:+.3f}")
        columns[2].metric("Subjectivity", f"{sentiment['textblob_subjectivity']:.2f}")
        st.info(
            "**Sentiment is not the disaster prediction.** It measures emotional tone, "
            "which is a different question. A factual disaster report "
            "(*\"7.8 magnitude earthquake strikes Nepal\"*) often reads as neutral, while "
            "an angry tweet about traffic reads as strongly negative without describing "
            "any emergency. It is shown here for context only - it does not feed the "
            "classification above.",
            icon="💡",
        )


def render_classifier(model) -> None:
    """The main input + predict area."""
    st.markdown('<p class="section-title">🔎 Classify a Tweet</p>', unsafe_allow_html=True)
    st.markdown(
        "Enter a tweet below and Digo will predict whether it refers to a **real "
        "disaster** or uses disaster language in an everyday, figurative way."
    )

    st.markdown("**Try an example:**")
    columns = st.columns(len(EXAMPLE_TWEETS))
    for column, (name, example) in zip(columns, EXAMPLE_TWEETS.items()):
        column.button(
            name,
            use_container_width=True,
            on_click=_use_example,
            args=(example,),
            key=f"example_{name}",
        )

    tweet = st.text_area(
        "Tweet text",
        key="tweet_input",
        height=130,
        max_chars=MAX_INPUT_CHARS,
        placeholder="e.g. Forest fire near La Ronge Sask. Canada",
        help="Paste or type a tweet, then click Classify.",
    )

    characters = len(tweet.strip()) if tweet else 0
    st.caption(f"{characters}/{MAX_INPUT_CHARS} characters")

    left, right, _ = st.columns([1, 1, 2])
    classify = left.button("🔍 Classify tweet", type="primary", use_container_width=True)
    right.button("Clear", use_container_width=True, on_click=_clear_input)

    if classify:
        is_valid, message = validate_input(tweet)
        if not is_valid:
            st.warning(message, icon="⚠️")
        else:
            try:
                result = predict_tweet(model, tweet.strip())
                render_result(result)
                log_prediction(tweet.strip(), result["label"], result["probability"])
            except Exception as error:  # never show a traceback to a normal user
                log_error("Prediction failed", error)
                st.error(
                    "Something went wrong while classifying that tweet. "
                    "Please try again, or check that the model file is valid.",
                    icon="🚫",
                )


def render_methodology(metadata: dict) -> None:
    """Methodology and limitations, below the fold."""
    st.markdown('<p class="section-title">📚 Methodology & Limitations</p>',
                unsafe_allow_html=True)

    with st.expander("⚙️  Methodology — how this works"):
        vectorizer = metadata.get("vectorizer", {})
        vocabulary = vectorizer.get("vocabulary_size")

        # Sizes come from the metadata the notebook wrote, so this text cannot
        # drift away from the artifacts actually shipped.
        bert = metadata.get("distilbert_comparison", {})
        bert_mb = bert.get("weights_size_mb")
        shipped_mb = metadata.get("artifact_size_mb")
        bert_text = f"roughly {bert_mb:.0f} MB" if bert_mb else "far larger"
        shipped_text = f"~{shipped_mb:.0f} MB" if shipped_mb else "much smaller"

        st.markdown(
            f"""
**1. Preprocessing.** The tweet is cleaned by the same `clean_text` function used during
training: HTML entities are unescaped, URLs and @mentions are replaced with placeholder
tokens, hashtag words are kept (`#CaliforniaWildfire` → `california wildfire`), and
common words are removed. **Negation words are deliberately kept**, because *"no
flooding"* means the opposite of *"flooding"*.

**2. Vectorization.** The cleaned text is converted to **TF-IDF** features - each term is
weighted by how often it appears in this tweet against how rare it is across the whole
corpus. The fitted vocabulary holds {f"{vocabulary:,}" if vocabulary else "n/a"} terms.

**3. Classification.** A linear Support Vector classifier draws a boundary in that
high-dimensional space, wrapped in probability calibration so the confidence shown above
is a real probability rather than an arbitrary score.

**4. Why not a larger model?** DistilBERT was fine-tuned during development and is
reported in the notebook. It is {bert_text} - too large for this deployment - so the
{shipped_text} classical pipeline ships instead. The trade-off is documented rather than
hidden.
"""
        )

    bert = metadata.get("distilbert_comparison", {})
    metrics = metadata.get("test_metrics", {})
    if bert and metrics:
        with st.expander("📊  Model comparison — deployed model vs DistilBERT"):
            st.markdown(
                "Both were evaluated on the **same held-out test set**. DistilBERT was "
                "genuinely fine-tuned during development — these are real measured "
                "results, not quoted figures."
            )
            st.markdown(
                f"""
| Metric | {metadata.get('model_name', 'Deployed model')} *(deployed)* | DistilBERT *(not deployed)* |
|---|---|---|
| Accuracy | {metrics['accuracy']:.4f} | **{bert['accuracy']:.4f}** |
| F1 (disaster) | {metrics['f1']:.4f} | **{bert['f1']:.4f}** |
| Macro-F1 | {metrics['macro_f1']:.4f} | **{bert['macro_f1']:.4f}** |
| ROC-AUC | {metrics['roc_auc']:.4f} | **{bert['roc_auc']:.4f}** |
| Artifact size | **{metadata.get('artifact_size_mb', 0):.2f} MB** | ~{bert.get('weights_size_mb', 0):.0f} MB |
| Training time | seconds | {bert.get('training_minutes', 0):.1f} min (CPU) |
"""
            )
            st.info(
                "**The transformer is the better model** — it is not deployed only "
                "because its size exceeds GitHub's 100 MB per-file limit and its "
                "inference is far slower on a free hosting tier. This is a documented "
                "trade-off, not a claim that the two models are equivalent.",
                icon="⚖️",
            )

    with st.expander("⚠️  Limitations — please read"):
        st.markdown(
            """
- **This is not an emergency service.** Never rely on this tool for safety decisions.
  It classifies *language*; it cannot verify that an event actually happened.
- **Negated statements are misclassified.** *"There is no flooding in our area"* is
  predicted as disaster-related. This is a measured weakness: the training data contains
  very few negated disaster statements, so the model learns a strong "flooding implies
  disaster" association it cannot override. A fine-tuned DistilBERT made the same
  mistake during development, so this is a data limitation, not a quirk of this model.
- **English only.** No Hindi, Marathi or Gujarati support is implemented.
- **Trained on 2015-era tweets.** Vocabulary, slang and events have moved on.
- **Sarcasm and irony are hard.** Figurative language is the known weak point.
- **Short inputs are unreliable.** A one-word tweet gives the model very little to use.
- **Human labels disagree.** In the source data, 18 identical tweets were labelled both
  ways by human annotators - so some error is irreducible.
- **No disaster-type detection.** The dataset provides no reliable type label, so the app
  does not claim to identify *which kind* of disaster.
"""
        )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model, load_error = load_model()
    metadata = load_metadata()

    render_hero()
    render_sidebar(metadata)
    render_about_panels(metadata)

    if load_error:
        st.error(load_error, icon="🚫")
        st.stop()

    st.divider()
    render_classifier(model)
    st.divider()
    render_methodology(metadata)

    st.markdown(
        f"""
        <div class="digo-footer">
            <strong>{APP_TITLE}</strong><br>
            {STUDENT['project_number']} · {STUDENT['programme']} · Built by {STUDENT['name']}<br>
            Built with scikit-learn, VADER, TextBlob and Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
