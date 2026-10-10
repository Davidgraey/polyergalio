"""Span labeling editor (drag-to-select picker with a slider fallback)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from components.span_picker import available as span_picker_available, span_picker
from spans import add_span, normalize_range
from tokenizing import tokenize

from .base import (
    PREVIEW_HEIGHT,
    Ctx,
    badge,
    color_for,
    confidence_slider,
    header,
    highlight,
    plain,
    set_conf,
    text_card,
)
from .cards import distribution_card, history_card, prediction_card


def span_body(ctx: Ctx) -> None:
    left, right = st.columns([3, 2])
    with left:
        text_card("Spans in text", highlight(ctx.sample.text, ctx.draft["value"]["spans"], ctx.task.labels),
                  height=PREVIEW_HEIGHT)
        span_editor(ctx)
    with right:
        span_label_card(ctx)
        prediction_card(ctx)
        distribution_card(ctx)
        history_card(ctx)


def span_editor(ctx: Ctx) -> None:
    text, labels = ctx.sample.text, ctx.task.labels
    toks = tokenize(text)
    spans = ctx.draft["value"]["spans"]
    with st.container(border=True):
        st.subheader("Spans")
        if not toks:
            st.caption("This row has no tokens.")
            return
        a, b = normalize_range(ctx.view.get("range"), len(toks))
        drag = span_picker_available()
        if drag:
            def set_drag(flag_key=ctx.key("drag")) -> None:
                ctx.view["drag"] = bool(st.session_state[flag_key])

            drag = st.toggle("Drag to select", value=ctx.view.get("drag", True), key=ctx.key("drag"),
                             on_change=set_drag, help="Select by dragging over the text. Off uses a slider.")
        if drag:
            picked = span_picker(f"picker::{ctx.task.id}::{ctx.sample.id}", text, toks, spans,
                                 lambda lab: color_for(labels, lab), (a, b))
            if picked:
                ctx.view["range"] = normalize_range(picked, len(toks))
                a, b = ctx.view["range"]
        elif len(toks) > 1:
            rk = ctx.key("range")

            def set_range() -> None:
                ctx.view["range"] = normalize_range(st.session_state[rk], len(toks))

            st.select_slider("Start and end", list(range(len(toks))), value=(a, b), key=rk,
                             format_func=lambda i: toks[i][0], on_change=set_range)
        s, e = toks[a][1], toks[b][2]
        st.caption(f"Selected: {plain(text[s:e])}")

        tk = ctx.key("type")
        kind = ctx.view.get("span_type")

        def set_type() -> None:
            ctx.view["span_type"] = st.session_state[tk]

        st.segmented_control("Type", labels, default=kind if kind in labels else None, key=tk,
                             on_change=set_type, label_visibility="collapsed")
        ck = ctx.key("span_conf")

        def set_span_conf() -> None:
            ctx.view["conf"] = float(st.session_state[ck])

        confidence_slider("Confidence", ctx.view.get("conf", 1.0), ck, set_span_conf)

        def selected() -> dict:
            label = ctx.view.get("span_type")
            if label not in labels:
                raise ValueError("Choose a span type.")
            x, y = normalize_range(ctx.view.get("range"), len(toks))
            return {"start": toks[x][1], "end": toks[y][2], "text": text[toks[x][1]:toks[y][2]],
                    "label": label, "confidence": float(ctx.view.get("conf", 1.0)),
                    "token_start": x, "token_end": y}

        def guarded_edit(fn) -> None:
            try:
                span = selected()
            except ValueError as err:
                ctx.notify("warning", str(err))
                return

            def apply(d: dict) -> None:
                d["value"]["spans"] = fn(d["value"]["spans"], span)
            try:
                ctx.edit(apply, bump=True)
                ctx.view["pick"] = None
            except ValueError as err:
                ctx.notify("warning", str(err))

        picked = ctx.view.get("pick")
        if picked is not None and not 0 <= picked < len(spans):
            picked = ctx.view["pick"] = None

        def add() -> None:
            guarded_edit(lambda ex, sp: add_span(ex, sp))

        def update() -> None:
            index = ctx.view.get("pick")
            guarded_edit(lambda ex, sp: add_span(ex, sp, skip=index))

        def delete() -> None:
            index = ctx.view.get("pick")
            ctx.edit(lambda d: d["value"]["spans"].pop(index), bump=True)
            ctx.view["pick"] = None

        with st.container(horizontal=True):
            st.button("Add span", icon=":material/add:", key=f"{ctx.task.id}::add", on_click=add)
            st.button("Update", icon=":material/edit:", key=f"{ctx.task.id}::update", on_click=update,
                      disabled=picked is None)
            st.button("Delete", icon=":material/delete:", key=f"{ctx.task.id}::delete", on_click=delete,
                      disabled=picked is None)
        if spans:
            pk = ctx.key("pick")

            def pick() -> None:
                index = st.session_state[pk]
                ctx.view["pick"] = index
                if index is not None:
                    sp = spans[index]
                    ctx.view["range"] = (sp.get("token_start", 0), sp.get("token_end", 0))
                    ctx.view["span_type"] = sp["label"]
                    ctx.view["conf"] = float(sp["confidence"])
                ctx.bump()

            st.pills("Spans", list(range(len(spans))), default=picked, key=pk, on_change=pick,
                     label_visibility="collapsed",
                     format_func=lambda i: badge(f"{spans[i]['text']} {float(spans[i]['confidence']):.2f}",
                                                 spans[i]["label"], color_for(labels, spans[i]["label"])))
            st.caption("Pick a span to move its start and end, then update it.")


def span_label_card(ctx: Ctx) -> None:
    spans = ctx.draft["value"]["spans"]
    with st.container(border=True):
        header("Label", ctx)
        if spans:
            confs = [float(sp["confidence"]) for sp in spans]
            st.write(f"{len(spans)} span(s) · mean confidence {np.mean(confs):.2f}")
        else:
            st.caption("No spans yet. If the text has none, set your confidence and save.")
            ck = ctx.key("none_conf")
            confidence_slider("Confidence that there are no spans", ctx.draft["confidence"], ck,
                              set_conf, (ctx, ck))
