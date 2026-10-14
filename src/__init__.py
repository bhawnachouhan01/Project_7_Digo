"""Digo - NLP Disaster Tweet Classification.

Reusable source modules shared by the training notebook, the automated tests
and the Streamlit application. Importing the project's logic from here
(rather than copying it) is what guarantees that a tweet typed into the app
is processed identically to the tweets the model was trained on.

Modules
-------
config
    Paths, random seed, split sizes, label names. All paths are relative to
    the repository root so the project runs unchanged on Linux.
preprocessing
    Text cleaning and tokenization. Negation-preserving, no runtime downloads.
features
    Engineered numeric features measured from the *raw* tweet.
sentiment
    VADER + TextBlob sentiment scoring.
modeling
    scikit-learn pipeline factories (picklable transformers live here).
evaluation
    Metrics, plots and error analysis.
logger
    Optional file logging for the app.
"""

__version__ = "1.0.0"
__project__ = "Digo - NLP Disaster Tweet Classification"
