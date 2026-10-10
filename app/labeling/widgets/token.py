"""Token labeling editor."""

from __future__ import annotations

from typing import List

import streamlit as st

from spans import tags_to_spans
from task_defs import OUTSIDE_TAG

from .base import (
    CONTEXT_HEIGHT,
    PREVIEW_HEIGHT,
    Ctx,
    color_for,
    confidence_slider,
    header,
    highlight,
    plain,
    text_card,
)
from .cards import distribution_card, history_card, prediction_card


def token_body(ctx: Ctx) -> None:
    left, right = st.columns([3, 2])
    with left:
        token_selector(ctx)
        text_card("Tagged text", highlight(ctx.sample.text, tags_to_spans(ctx.draft["value"]["tokens"]),
                                           ctx.task.labels), height=PREVIEW_HEIGHT)
        prediction_card(ctx)
    with right:
        token_tag_card(ctx)
        distribution_card(ctx)
        history_card(ctx)


def token_pill(tok: dict, labels: List[str]) -> str:
    tag = tok.get("tag") or OUTSIDE_TAG
    if tag == OUTSIDE_TAG:
        return plain(tok["text"])
    conf = float(tok.get("confidence", 1.0))
    suffix = f" {conf:.2f}" if conf < 1.0 else ""
    return f"{plain(tok['text'])} :{color_for(labels, tag)}-badge[{plain(tag)}{suffix}]"


def token_selector(ctx: Ctx) -> None:
    toks = ctx.draft["value"]["tokens"]
    with st.container(border=True):
        st.subheader("Tokens")
        st.caption("Select tokens, then pick a tag on the right.")
        with st.container(height=CONTEXT_HEIGHT):
            if not toks:
                st.caption("This row has no tokens.")
                return
            key = ctx.key("selection")

            def set_selection() -> None:
                ctx.view["selection"] = sorted(int(i) for i in st.session_state[key] or [])

            st.pills("Tokens", list(range(len(toks))), selection_mode="multi", key=key,
                     default=ctx.view.get("selection", []),
                     format_func=lambda i: token_pill(toks[i], ctx.task.labels),
                     on_change=set_selection, label_visibility="collapsed")


def token_tag_card(ctx: Ctx) -> None:
    labels = ctx.task.labels
    selection = ctx.view.get("selection", [])
    toks = ctx.draft["value"]["tokens"]
    with st.container(border=True):
        header("Tags", ctx)
        st.caption(f"{len(selection)} token(s) selected. The tag and confidence apply to the selection.")
        ck = ctx.key("tag_conf")

        def set_tag_conf() -> None:
            ctx.view["conf"] = float(st.session_state[ck])

        confidence_slider("Confidence for applied tags", ctx.view.get("conf", 1.0), ck, set_tag_conf)

        def apply(tag: str) -> None:
            chosen, conf = list(ctx.view.get("selection", [])), float(ctx.view.get("conf", 1.0))

            def fn(d: dict) -> None:
                for i in chosen:
                    d["value"]["tokens"][i].update(tag=tag, confidence=conf)
            ctx.edit(fn, bump=True)
            ctx.view["selection"] = []

        with st.container(horizontal=True, gap="small"):
            for i, tag in enumerate(labels):
                text = plain(tag) if tag == OUTSIDE_TAG else f":{color_for(labels, tag)}[{plain(tag)}]"
                st.button(text, key=f"{ctx.task.id}::tag::{i}", on_click=apply, args=(tag,),
                          disabled=not selection)

        def select_untagged() -> None:
            ctx.view["selection"] = [i for i, t in enumerate(toks) if (t.get("tag") or OUTSIDE_TAG) == OUTSIDE_TAG]
            ctx.bump()

        def select_low() -> None:
            ctx.view["selection"] = [i for i, t in enumerate(toks) if float(t.get("confidence", 1.0)) < 1.0]
            ctx.bump()

        def clear() -> None:
            ctx.view["selection"] = []
            ctx.bump()

        with st.container(horizontal=True):
            st.button("Select untagged", icon=":material/select_all:", key=f"{ctx.task.id}::sel_o",
                      on_click=select_untagged)
            st.button("Select low confidence", icon=":material/help:", key=f"{ctx.task.id}::sel_low",
                      on_click=select_low)
            st.button("Clear", icon=":material/deselect:", key=f"{ctx.task.id}::sel_clear",
                      on_click=clear, disabled=not selection)
