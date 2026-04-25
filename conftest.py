"""Root conftest.py — adds the project root to sys.path so that bare
imports like 'from models import ...' resolve correctly when running pytest
from the project root.
"""
import sys
from pathlib import Path

# Ensure the project root is the first entry on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
