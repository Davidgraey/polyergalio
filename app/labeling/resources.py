"""Cached resources: the parsed dataset and the one label store per project."""

from __future__ import annotations

import streamlit as st

from dataset import load_dataset
from label_store import LabelStore


@st.cache_data(show_spinner=False)
def cached_dataset(path: str, mtime_ns: int, size: int):
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def get_store(root: str) -> LabelStore:
    """One store per project, shared by every session of this server."""
    return LabelStore(root)
