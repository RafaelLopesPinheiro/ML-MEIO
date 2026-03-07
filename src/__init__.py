"""
src/__init__.py — Package initializer.

Adds the project root to sys.path so that all modules inside src/
can do `import config` and find config.py at the project root level.
"""

import sys
from pathlib import Path

# Project root is one level up from src/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)