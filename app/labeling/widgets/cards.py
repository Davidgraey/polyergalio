"""Side cards shared by every task editor: guidelines, model prediction, label distribution and history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from label_events import summarize_value
from spans import tags_to_spans
from task_defs import TaskDef, TaskType

from .base import CHART_HEIGHT, Ctx, color_for, highlight, plain


GUIDANCE_HEIGHT = 330


def guidance_card(task: TaskDef) -> None:
    """Rubric (labeling instructions) and per-label definitions, shown beside the labeling controls."""
    with st.container(border=True, key=f"guidance_{task.id}"):
        st.markdown("##### :material/menu_book: Guidelines")
        with st.container(height=GUIDANCE_HEIGHT, border=False):
            if task.rubric.strip():
                st.markdown(plain(task.rubric))
            else:
                st.caption("No rubric has been written for this task.")
            st.markdown("**Label definitions**")
            for label in task.labels:
                meaning = task.definition_of(label)
                st.markdown(f":{color_for(task.labels, label)}-badge[{plain(label)}] "
                            + (plain(meaning) if meaning else "*No definition yet.*"))


def prediction_card(ctx: Ctx) -> None:
    pred, task = ctx.pred, ctx.task
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown("**Model**")
            st.badge(ctx.model_label, color="gray")
            for name, v in ctx.scores.items():
                short = name.split(" (")[0].replace("Least confidence", "Least conf.")
                st.badge(f"{short} {v:.2f}", color="orange" if v >= 0.66 else "gray")
        t = task.type
        if t is TaskType.CLASSIFICATION:
            st.bar_chart(pd.DataFrame({"name": pred.labels, "probability": np.asarray(pred.probs, float)}),
                         x="name", y="probability", sort=False, height=CHART_HEIGHT)
        elif t is TaskType.TOKEN and pred.probs.size:
            top = pred.argmax_labels()
            pseudo = [{"i": i, "start": s, "end": e, "tag": top[i]} for i, (_, s, e) in enumerate(pred.tokens)]
            st.markdown(highlight(ctx.sample.text, tags_to_spans(pseudo), task.labels) or "*No entities predicted.*")
        elif t is TaskType.SPAN:
            st.markdown(highlight(ctx.sample.text, pred.suggested_spans(ctx.sample.text), task.labels)
                        or "*No spans predicted.*")
        elif t is TaskType.CONTRASTIVE and pred.candidate_ids:
            st.dataframe(pd.DataFrame({"candidate": pred.candidate_ids,
                                       f"p({task.labels[0]})": np.round(pred.probs, 3)}),
                         hide_index=True, width="stretch")


def distribution_card(ctx: Ctx) -> None:
    counts = ctx.extra.get("distribution")
    with st.expander("Label distribution", icon=":material/bar_chart:"):
        if counts is None or counts.empty or counts["count"].sum() == 0:
            st.caption("No labels yet.")
            return
        st.bar_chart(counts, x="name", y="count", sort=False, height=CHART_HEIGHT)
        if ctx.extra.get("mean_conf") is not None:
            st.caption(f"Mean confidence {ctx.extra['mean_conf']:.2f} across saved labels.")


def history_card(ctx: Ctx) -> None:
    history = ctx.extra.get("history") or []
    with st.expander(f"History ({len(history)})", icon=":material/history:"):
        if not history:
            st.caption("No label history yet.")
            return
        st.dataframe(pd.DataFrame([{
            "seq": h["seq"], "time": h["ts"][:19].replace("T", " "), "op": h["op"],
            "annotator": h["annotator"], "mode": h["mode"], "confidence": h["confidence"],
            "value": summarize_value(h["task_type"], h["value"]),
        } for h in reversed(history)]), hide_index=True, width="stretch")
