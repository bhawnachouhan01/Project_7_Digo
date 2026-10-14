"""Shared pytest configuration.

Puts the project root on ``sys.path`` so that ``import src...`` and ``import app``
work when pytest is run from anywhere in the project.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
