"""Labeling tab: the sample panel, the labeling panel with its single action bar, and the open task's editor."""

from __future__ import annotations

import numpy as np
import streamlit as st

import widgets as W
from label_events import summarize_value
from label_stats import distribution
from review_queue import needs_work
from session import bump_version, flash, guarded, notify, show_notice, ss, version, view_state
from task_defs import Mode, TaskType
from task_session import TaskSession
from widgets import BODIES, TYPE_ICONS, Ctx, plain


def render_sample(app, active, tab_labels, opts, order, score_lookup, top_task, labeled) -> None:
    ds, mode = app.ds, app.mode
    nav_key = f"nav::{mode.value}"
    sid = ss.get(nav_key)
    if sid not in order:
        pos = min(int(ss.get(nav_key + "::pos", 0)), len(order) - 1)
        sid = order[pos]
    pos = order.index(sid)
    ss[nav_key], ss[nav_key + "::pos"] = sid, pos
    sample = ds.by_id[sid]
    sessions = {t.id: TaskSession(app, t, sample) for t in active}
    strategy = opts.get("strategy")

    prev_sid = order[pos - 1] if pos > 0 else None
    next_sid = order[pos + 1] if pos + 1 < len(order) else None
    after = next_sid or prev_sid or sid
    rest = order[pos + 1:] + order[:pos]
    next_incomplete = next((i for i in rest if any(needs_work(app, t, i) for t in active)), None)

    if ss.get("task_tab") not in tab_labels.values():
        ss.pop("task_tab", None)
    task = next((t for t in active if tab_labels[t.id] == ss.get("task_tab")), active[0])
    s = sessions[task.id]

    def go(target) -> None:
        if target is None:
            return
        ss[nav_key] = target
        for t in active:
            view = view_state(t.id)
            keep = view.get("conf")
            view.clear()
            if keep is not None:
                view["conf"] = keep
            bump_version(t.id)
        if mode is Mode.RELABEL and target in top_task:
            ss["task_tab"] = tab_labels[top_task[target]]
        elif mode is Mode.INITIAL:
            first = next((t for t in active if needs_work(app, t, target)), None)
            if first is not None:
                ss["task_tab"] = tab_labels[first.id]

    def save_all() -> None:
        saved, failed = [], []
        for t in active:
            if not sessions[t.id].has_draft():
                continue
            errs = sessions[t.id].commit(strategy)
            (failed.append(f"{t.name}: {' '.join(errs[:3])}") if errs else saved.append(t.name))
        if failed:
            notify("page", "error", ("Saved " + ", ".join(saved) + ". " if saved else "")
                   + "Not saved, fix and try again: " + " | ".join(failed))
            return
        flash(f"Saved {', '.join(saved)} for {sid}" if saved else f"No edits on {sid}, moved on")
        go(after)

    def save_task() -> None:
        errs = s.commit(strategy)
        if errs:
            notify(task.id, "error", "Not saved. " + " ".join(errs[:5]))
            return
        flash(f"Saved {task.name} for {sid}")
        bump_version(task.id)
        i = [t.id for t in active].index(task.id)
        following = [active[(i + k) % len(active)] for k in range(1, len(active))]
        nxt = next((t for t in following if needs_work(app, t, sid)), None)
        if nxt is not None:
            ss["task_tab"] = tab_labels[nxt.id]
        else:
            go(after)

    def discard() -> None:
        s.discard()
        view_state(task.id).pop("selection", None)
        view_state(task.id).pop("pick", None)

    def remove() -> None:
        s.remove()
        flash(f"Removed the {task.name} label for {sid}. It stays in the history.")

    with st.container(border=True, key="sample_panel"):
        done = sum(1 for t in active if sessions[t.id].cur is not None)
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"##### :material/article: SAMPLE · {plain(sid)}")
            st.badge(f"{pos + 1} of {len(order)}", color="gray")
            if done == len(active):
                st.badge("complete", color="green", icon=":material/task_alt:")
            else:
                st.badge(f"{done} of {len(active)} tasks labeled", color="gray")
            if sid in score_lookup:
                driver = next((t.name for t in active if t.id == top_task.get(sid)), "")
                st.badge(f"queue score {score_lookup[sid]:.2f} · {driver}", color="orange",
                         icon=":material/query_stats:")
        with st.container(border=True, key="sample_text"):
            st.caption("TEXT TO LABEL")
            with st.container(height=W.CONTEXT_HEIGHT, border=False):
                st.markdown(plain(sample.text))
        with st.container(horizontal=True, gap="small"):
            for t in active:
                label, color, _ = W.STATUS_BADGE[sessions[t.id].status]
                st.badge(f"{t.name}: {label}", color=color, icon=TYPE_ICONS[t.type])

    with st.container(border=True, key="label_panel"):
        title = "RELABELING" if mode is Mode.RELABEL else "LABELING"
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key="action_bar"):
            st.markdown(f"##### :material/edit_note: {title}")
            st.button("", icon=":material/chevron_left:", key="prev", on_click=go, args=(prev_sid,),
                      disabled=prev_sid is None, help="Previous sample")
            st.button("", icon=":material/chevron_right:", key="next", on_click=go, args=(next_sid,),
                      disabled=next_sid is None, help="Next sample")
            st.button("", icon=":material/skip_next:", key="next_incomplete", on_click=go,
                      args=(next_incomplete,), disabled=next_incomplete is None,
                      help="Next sample with unfinished tasks")
            st.button(f"Save {task.name}", icon=":material/check:", key="save_task", on_click=save_task,
                      disabled=not app.annotator, help="Save this task, then open the next unfinished one.")
            st.button("Save all and next", icon=":material/done_all:", type="primary", key="save_all",
                      on_click=save_all, disabled=not app.annotator,
                      help="Save every task with edits on this sample, then move to the next sample.")
            st.button("Discard", icon=":material/restart_alt:", type="tertiary", key="discard",
                      on_click=discard, disabled=s.stored is None, help=f"Discard unsaved {task.name} edits")
            st.button("Remove", icon=":material/delete:", type="tertiary", key="remove", on_click=remove,
                      disabled=s.cur is None or not app.annotator,
                      help=f"Remove the saved {task.name} label (kept in history)")
        if not app.annotator:
            st.caption("Set an annotator name in the sidebar to save.")
        show_notice("notice::page")

        tabs = st.tabs(list(tab_labels.values()), key="task_tab", on_change="rerun")
        for tab, t in zip(tabs, active):
            if tab.open:
                with tab:
                    guarded(render_task_editor, app, t, sessions[t.id], labeled)


def render_task_editor(app, task, s: TaskSession, labeled) -> None:
    show_notice(f"notice::{task.id}")
    view = view_state(task.id)
    sid = s.sample.id
    if app.mode is Mode.RELABEL and s.cur:
        st.caption(f"Current: {plain(summarize_value(task.type.value, s.cur['value']))} · confidence "
                   f"{s.cur['confidence']:.2f} · {plain(s.cur['annotator'])}. Saving adds a new version.")
    if task.type is TaskType.SPAN and "span_type" not in view and task.labels:
        view["span_type"] = task.labels[0]

    confs = [r["confidence"] for r in labeled[task.id].values()]
    ver = version(task.id)
    ctx = Ctx(
        task=task, sample=s.sample, current=s.cur, draft=s.draft, status=s.status, pred=s.pred,
        scores=s.scores, candidates=s.candidates or [], view=view,
        model_label=f"{app.model_key} v{app.model_version}",
        key=lambda base: f"w::{task.id}::{sid}::{ver}::{int(app.prefill)}{app.model_version}::{base}",
        edit=s.edit, bump=lambda: bump_version(task.id),
        notify=lambda level, text: notify(task.id, level, text),
        extra={"distribution": distribution(task, labeled[task.id]),
               "mean_conf": float(np.mean(confs)) if confs else None,
               "history": app.store.history(sid, task.id)},
    )
    work, guide = st.columns([5, 2])
    with work:
        BODIES[task.type](ctx)
    with guide:
        W.guidance_card(task)
