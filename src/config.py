"""Central configuration for the Digo disaster-tweet classification project.

Every path in this project is derived from :data:`PROJECT_ROOT`, which is
resolved *relative to this file*. Nothing here contains a Windows drive letter
or an absolute user path, so the same code runs unchanged on Streamlit
Community Cloud (Linux) after a ``git clone``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# src/config.py  ->  src/  ->  project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

RAW_DATA_FILE: Path = DATA_DIR / "twitter_disaster.csv"

MODEL_FILE: Path = MODELS_DIR / "selected_model.joblib"
METADATA_FILE: Path = MODELS_DIR / "model_metadata.json"
COMPARISON_FILE: Path = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Label semantics
# ---------------------------------------------------------------------------
# The positive class (1) is "disaster-related". This matters for precision /
# recall / ROC-AUC, which are all reported with respect to class 1.
POSITIVE_CLASS: int = 1
LABEL_NAMES: dict[int, str] = {0: "Non-disaster", 1: "Disaster-related"}

# ---------------------------------------------------------------------------
# Input limits used by the Streamlit app
# ---------------------------------------------------------------------------
# The longest tweet in the supplied dataset is 157 characters. We allow a
# generous 500 so users can paste longer modern posts, while still rejecting
# pasted articles that are far outside the training distribution.
MIN_INPUT_CHARS: int = 3
MAX_INPUT_CHARS: int = 500


def ensure_directories() -> None:
    """Create the output directories if they do not already exist.

    Safe to call repeatedly. Only creates *output* folders; it never touches
    the dataset.
    """
    for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
