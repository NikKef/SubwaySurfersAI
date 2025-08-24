"""Pytest configuration for package imports."""

import pathlib
import sys

# Ensure the project root is on sys.path so ``import src`` works without installing the package.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
