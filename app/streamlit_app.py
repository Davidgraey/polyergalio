"""
Streamlit app for the polyergalio examples.

Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for folder in (APP_DIR.parent / "src", APP_DIR.parent / "examples", APP_DIR):
    sys.path.insert(0, str(folder))

import clustering_layers_tab
import clustering_tab
import moe_routing_tab
import orchestration_tab
import relative_weights_tab
import scg_regression_tab
import spectre_tab
import streamlit as st
import system_one_tab
import text_distortions_tab
import tree_models_tab

CATEGORIES = {
    "Supervised": {
        "Relative weights": relative_weights_tab.render,
        "SCG regression": scg_regression_tab.render,
        "Boosted trees": tree_models_tab.render,
    },
    "Clustering": {
        "Self-organizing maps": clustering_tab.render,
    },
    "Neural networks": {
        "System one decision": system_one_tab.render,
        "Spectre encoder-decoder": spectre_tab.render,
        "MoE routing": moe_routing_tab.render,
        "Clustering layers": clustering_layers_tab.render,
        "Distributed training": orchestration_tab.render,
    },
    "Text": {
        "Token distortions": text_distortions_tab.render,
    },
}


def main() -> None:
    st.set_page_config(page_title="polyergalio examples", layout="wide")
    st.title("polyergalio examples")
    for tab, examples in zip(st.tabs(list(CATEGORIES)), CATEGORIES.values()):
        with tab:
            for example_tab, render in zip(st.tabs(list(examples)), examples.values()):
                with example_tab:
                    render()


if __name__ == "__main__":
    main()
