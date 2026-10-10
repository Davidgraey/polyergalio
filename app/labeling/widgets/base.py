"""Shared UI constants, text helpers, the editor context and small cards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from prediction import Prediction
from task_defs import OUTSIDE_TAG, TaskDef, TaskType


CONTEXT_HEIGHT = 150
PREVIEW_HEIGHT = 96
CHART_HEIGHT = 110
MAX_SEGMENTED = 7
COLORS = ["blue", "green", "orange", "violet", "red", "yellow", "gray"]


TYPE_ICONS = {
    TaskType.CLASSIFICATION: ":material/rule:",
    TaskType.TOKEN: ":material/sell:",
    TaskType.SPAN: ":material/segment:",
    TaskType.CONTRASTIVE: ":material/compare_arrows:",
}


_PUNCT = re.compile(r"([!-/:-@\[-`{-~])")


def plain(value: Any) -> str:
    """A value as Markdown-safe text (ASCII punctuation escaped)."""
    text = "-" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return _PUNCT.sub(r"\\\1", text)


def color_for(labels: List[str], label: Any) -> str:
    named = [x for x in labels if x != OUTSIDE_TAG]
    return COLORS[named.index(label) % len(COLORS)] if label in named else "gray"


def badge(text: str, label: str, color: str) -> str:
    return f":{color}-badge[{plain(text)} · {plain(label)}]"


def highlight(text: str, spans: Optional[List[dict]], labels: List[str]) -> str:
    """Text with character spans rendered as colored badges."""
    parts, cur = [], 0
    for sp in sorted(spans or [], key=lambda s: (int(s["start"]), -int(s["end"]))):
        s, e = int(sp["start"]), int(sp["end"])
        if s > cur and text[cur:s].strip():
            parts.append(plain(text[cur:s].strip()))
        parts.append(badge(text[s:e], sp.get("label", ""), color_for(labels, sp.get("label"))))
        cur = max(cur, e)
    if text[cur:].strip():
        parts.append(plain(text[cur:].strip()))
    return " ".join(parts)


@dataclass
class Ctx:
    """Everything a task editor needs for one sample.

    Attributes
    ----------
    draft : dict
        ``{"value": ..., "confidence": float}``.
    status : str
        One of ``saved``, ``edited``, ``draft``, ``unsaved``.
    view : dict
        Transient UI state for this task.
    key : callable
        ``key(base)`` returns a versioned widget key.
    edit : callable
        ``edit(fn, bump=False)`` mutates the draft and persists it.
    """

    task: TaskDef
    sample: Any
    current: Optional[dict]
    draft: dict
    status: str
    pred: Prediction
    scores: Dict[str, float]
    candidates: List[Any]
    view: dict
    model_label: str
    key: Callable[[str], str]
    edit: Callable[..., None]
    bump: Callable[[], None]
    notify: Callable[[str, str], None]
    extra: Dict[str, Any] = field(default_factory=dict)


STATUS_BADGE = {
    "saved": ("saved", "green", ":material/check:"),
    "edited": ("edited", "orange", ":material/edit:"),
    "draft": ("draft", "blue", ":material/draft:"),
    "unsaved": ("unsaved", "gray", None),
}


def header(title: str, ctx: Optional[Ctx] = None, extra_badges=()) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(title)
        for text, color in extra_badges:
            st.badge(text, color=color)
        if ctx is not None:
            text, color, icon = STATUS_BADGE[ctx.status]
            st.badge(text, color=color, icon=icon)


def text_card(title: str, body: str, height: int = CONTEXT_HEIGHT, caption: str = "") -> None:
    with st.container(border=True):
        st.subheader(title)
        if caption:
            st.caption(caption)
        with st.container(height=height):
            st.markdown(body or "*No text.*")


def confidence_slider(label: str, value: float, key: str, on_change=None, args=()) -> None:
    st.slider(label, 0.0, 1.0, float(value), 0.05, key=key, on_change=on_change, args=args)


def set_conf(ctx: Ctx, key: str) -> None:
    ctx.edit(lambda d: d.update(confidence=float(st.session_state[key])))
