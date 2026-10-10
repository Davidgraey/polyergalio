"""Full-sample classification editor."""

from __future__ import annotations

import streamlit as st

from .base import MAX_SEGMENTED, Ctx, confidence_slider, header, set_conf
from .cards import distribution_card, history_card, prediction_card


def classification_body(ctx: Ctx) -> None:
    left, right = st.columns([3, 2])
    with left:
        classification_card(ctx)
        prediction_card(ctx)
    with right:
        distribution_card(ctx)
        history_card(ctx)


def classification_card(ctx: Ctx) -> None:
    t, value = ctx.task, ctx.draft["value"]
    with st.container(border=True):
        header("Label", ctx, [("multi-label", "blue")] if t.multi_label else [])
        key = ctx.key("label")
        if t.multi_label:
            def set_labels() -> None:
                ctx.edit(lambda d: d["value"].update(labels=list(st.session_state[key] or [])))
            st.pills("Labels", t.labels, selection_mode="multi", default=value.get("labels", []),
                     key=key, on_change=set_labels, label_visibility="collapsed")
        else:
            def set_label() -> None:
                ctx.edit(lambda d: d["value"].update(label=st.session_state[key]))
            current = value.get("label")
            if len(t.labels) > MAX_SEGMENTED:
                st.radio("Label", t.labels, index=t.labels.index(current) if current in t.labels else None,
                         key=key, on_change=set_label, label_visibility="collapsed")
            else:
                st.segmented_control("Label", t.labels, default=current if current in t.labels else None,
                                     key=key, on_change=set_label, width="stretch",
                                     label_visibility="collapsed")
        ck = ctx.key("conf")
        confidence_slider("Confidence", ctx.draft["confidence"], ck, set_conf, (ctx, ck))
