"""
Data labeling app: decision, token and sequence labeling, one tab per task.

Run: streamlit run app/labeling/label_app.py
"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
for folder in (APP_DIR.parents[1] / "src", APP_DIR):
    sys.path.insert(0, str(folder))

import decision_tab
import sequence_tab
import token_tab

TABS = {
    ":material/rule: Decision": decision_tab,
    ":material/sell: Token labeling": token_tab,
    ":material/segment: Sequence labeling": sequence_tab,
}


def guarded(draw) -> None:
    """Draw a section; a display error is shown in place and never touches the stored labels."""
    try:
        draw()
    except Exception as error:
        st.error(f"This section failed to draw; your labels are unchanged in the session. {error}", icon=":material/error:")
        st.exception(error)


def main() -> None:
    st.set_page_config(page_title="Labeling", page_icon=":material/label:", layout="wide")
    st.title("Labeling", icon=":material/label:")
    tabs = st.tabs(list(TABS), key="task_tab", on_change="rerun")
    for tab, module in zip(tabs, TABS.values()):
        if tab.open:
            with st.sidebar:
                guarded(module.sidebar)
            with tab:
                guarded(module.render)


if __name__ == "__main__":
    main()
