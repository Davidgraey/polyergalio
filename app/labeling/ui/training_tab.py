"""Training tab (placeholder)."""

from __future__ import annotations

import streamlit as st

from widgets import plain


def training_tab(app, active) -> None:
    """Placeholder: fine-tuning configuration for the model loaded for HITL."""
    with st.container(border=True, key="train_panel"):
        st.markdown("##### :material/model_training: Training")
        st.badge(f"loaded model: {plain(app.model_key)} v{app.model_version}", color="gray", icon=":material/memory:")
        st.info("Fine-tuning is not wired up yet. This tab will hold the hyperparameters and configuration used "
                "to fine-tune the model loaded for HITL on the saved labels (learning rate, epochs, batch size, "
                "train/validation split, per-task loss weights, confidence weighting, checkpointing), plus run "
                "history. Use Retrain in the sidebar for the mock model meanwhile.", icon=":material/construction:")
