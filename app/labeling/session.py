"""Streamlit session-state helpers: notices, per-task view state and widget versions, and the section guard."""

from __future__ import annotations

import streamlit as st


ss = st.session_state


def guarded(draw, *args):
    """Draw a section; a display error is shown in place and never touches saved labels."""
    try:
        return draw(*args)
    except Exception as error:
        if type(error).__name__ in ("RerunException", "StopException"):
            raise
        st.error(f"This section failed to draw. Saved labels are unaffected. {error}", icon=":material/error:")
        st.exception(error)
        return None


def flash(text: str) -> None:
    ss["_flash"] = text


def global_notice(level: str, text: str) -> None:
    ss["_notice"] = (level, text)


def view_state(task_id: str) -> dict:
    return ss.setdefault(f"view::{task_id}", {})


def version(task_id: str) -> int:
    return ss.get(f"ver::{task_id}", 0)


def bump_version(task_id: str) -> None:
    ss[f"ver::{task_id}"] = version(task_id) + 1


def notify(task_id: str, level: str, text: str) -> None:
    ss[f"notice::{task_id}"] = (level, text)


def show_notice(slot: str) -> None:
    message = ss.pop(slot, None)
    if message:
        level, text = message
        if level == "error":
            st.error(text, icon=":material/error:")
        elif level == "warning":
            st.warning(text, icon=":material/warning:")
        else:
            st.info(text, icon=":material/info:")
