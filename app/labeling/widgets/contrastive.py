"""Contrastive labeling editor (candidates vs. the anchor sample)."""

from __future__ import annotations

import streamlit as st

from .base import Ctx, badge, color_for, header, plain
from .cards import distribution_card, history_card, prediction_card


def contrastive_body(ctx: Ctx) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.caption("The sample above is the anchor. Compare each candidate to it.")
        candidate_cards(ctx)
    with right:
        contrastive_label_card(ctx)
        prediction_card(ctx)
        distribution_card(ctx)
        history_card(ctx)


def candidate_cards(ctx: Ctx) -> None:
    labels = ctx.task.labels
    pairs = {p["sample_id"]: (i, p) for i, p in enumerate(ctx.draft["value"]["pairs"])}
    if not ctx.candidates:
        st.caption("Not enough other samples to compare against.")
        return
    for j, cand in enumerate(ctx.candidates):
        idx, pair = pairs.get(cand.id, (None, {}))
        if idx is None:
            continue
        with st.container(border=True):
            rel = pair.get("relation")
            header(f"Candidate {j + 1}",
                   extra_badges=[(rel, color_for(labels, rel))] if rel else [("unlabeled", "gray")])
            st.caption(cand.id)
            st.markdown(plain(cand.text))
            rk, ck = ctx.key(f"rel::{cand.id}"), ctx.key(f"conf::{cand.id}")

            def set_rel(i=idx, k=rk) -> None:
                ctx.edit(lambda d: d["value"]["pairs"][i].update(relation=st.session_state[k]))

            def set_pair_conf(i=idx, k=ck) -> None:
                ctx.edit(lambda d: d["value"]["pairs"][i].update(confidence=float(st.session_state[k])))

            with st.container(horizontal=True, vertical_alignment="bottom"):
                st.segmented_control("Relation", labels, default=rel if rel in labels else None, key=rk,
                                     on_change=set_rel, label_visibility="collapsed")
                st.slider("Confidence", 0.0, 1.0, float(pair.get("confidence", 1.0)), 0.05, key=ck,
                          on_change=set_pair_conf)


def contrastive_label_card(ctx: Ctx) -> None:
    pairs = ctx.draft["value"]["pairs"]
    with st.container(border=True):
        header("Label", ctx)
        done = sum(1 for p in pairs if p.get("relation"))
        st.write(f"{done} of {len(pairs)} candidate(s) labeled.")
        if pairs:
            st.markdown(" ".join(
                badge(p["sample_id"], p["relation"], color_for(ctx.task.labels, p["relation"]))
                for p in pairs if p.get("relation")) or "*Nothing labeled yet.*")
