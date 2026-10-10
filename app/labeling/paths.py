"""Data locations. Override the data folder with HITL_DATA_DIR."""

from __future__ import annotations

import os


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.environ.get("HITL_DATA_DIR") or os.path.join(HERE, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")
