"""The page: summary strip and the four top-level tabs."""

from __future__ import annotations

import numpy as np
import streamlit as st

from review_queue import labeled_for, sample_queue, stable_seed
from session import guarded, show_notice
from task_defs import Mode
from ui.data_tab import data_tab
from ui.labeling_tab import render_sample
from ui.tasks_tab import tasks_tab
from ui.training_tab import training_tab


MAIN_TABS = [":material/database: Data", ":material/tune: Tasks", ":material/edit_note: Labeling",
             ":material/model_training: Training"]


def summary_strip(n: int, complete: int, partial: int, labels: int, confs, in_view: int, mode: Mode) -> None:
    """One line of progress badges plus a thin bar, instead of metric cards."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(f"{complete} / {n} complete · {100 * complete / max(n, 1):.0f}%", color="green",
                 icon=":material/task_alt:")
        st.badge(f"{partial} in progress", color="blue", icon=":material/pending:")
        st.badge(f"{labels} labels saved", color="gray", icon=":material/sell:")
        st.badge(f"mean confidence {np.mean(confs):.2f}" if confs else "no labels yet", color="gray",
                 icon=":material/speed:")
        st.badge(f"{in_view} in queue" if mode is Mode.RELABEL else f"{in_view} to label", color="orange",
                 icon=":material/filter_list:")
    st.progress(complete / max(n, 1))


def labeling_tab(app, active, tab_labels, opts, order, score_lookup, top_task, labeled) -> None:
    if not order:
        show_notice("notice::page")
        if app.mode is Mode.INITIAL:
            st.success("Every sample is complete for the active tasks. Switch to Relabel to review "
                       "uncertain labels, or turn off Hide complete samples.", icon=":material/task_alt:")
        else:
            st.info("The queue is empty for these settings. Try another scope, strategy or a lower "
                    "minimum score.", icon=":material/info:")
    else:
        render_sample(app, active, tab_labels, opts, order, score_lookup, top_task, labeled)


def render_page(app, active, tab_labels: dict, opts: dict) -> None:
    ds, mode = app.ds, app.mode
    labeled = {t.id: labeled_for(app, t) for t in active}
    complete = {s.id for s in ds.samples if all(s.id in labeled[t.id] for t in active)}
    touched = {sid for t in active for sid in labeled[t.id]}

    score_lookup, top_task = {}, {}
    if mode is Mode.INITIAL:
        order = [s.id for s in ds.samples]
        if opts.get("order") == "Shuffled":
            perm = np.random.default_rng(stable_seed(ds.sha256)).permutation(len(order))
            order = [order[i] for i in perm]
        if opts.get("hide", True):
            order = [i for i in order if i not in complete]
    else:
        order, score_lookup, top_task = sample_queue(app, active, opts)

    if app.issues:
        st.error(f"{len(app.issues)} saved label(s) failed verification. See Data > Integrity.",
                 icon=":material/error:")
    confs = [r["confidence"] for t in active for r in labeled[t.id].values()]
    summary_strip(len(ds.samples), len(complete), len(touched - complete),
                  sum(len(v) for v in labeled.values()), confs, len(order), mode)

    t_data, t_tasks, t_label, t_train = st.tabs(MAIN_TABS, key="main_tab", default=MAIN_TABS[2],
                                                on_change="rerun")
    if t_data.open:
        with t_data:
            guarded(data_tab, app, active, labeled, opts, complete)
    if t_tasks.open:
        with t_tasks:
            guarded(tasks_tab, app)
    if t_label.open:
        with t_label:
            labeling_tab(app, active, tab_labels, opts, order, score_lookup, top_task, labeled)
    if t_train.open:
        with t_train:
            guarded(training_tab, app, active)
