"""Tasks tab: set up, edit and configure labeling tasks."""

from __future__ import annotations

import streamlit as st

import widgets as W
from project_config import load_tasks, save_tasks
from session import flash, global_notice, ss
from task_admin import apply_task_edit, build_task
from task_defs import TaskType
from widgets import TYPE_ICONS, plain


def toggle_archive(root: str, task_id: str) -> None:
    tasks = load_tasks(root)
    for t in tasks:
        if t.id == task_id:
            t.archived = not t.archived
    save_tasks(root, tasks)


def add_task(root: str) -> None:
    tasks = load_tasks(root)
    new, msg = build_task(tasks, ss.get("new_name"), ss.get("new_type"), ss.get("new_labels"),
                          ss.get("new_multi"), ss.get("new_ncand"), ss.get("new_rubric"),
                          ss.get("new_label_col"), ss.get("new_pred_col"), ss.get("new_defs"))
    if new is None:
        global_notice("warning", msg)
        return
    tasks.append(new)
    save_tasks(root, tasks)
    flash(msg)


def save_task_edit(root: str, task_id: str) -> None:
    """Apply the edit form for one task. Existing labels can't be renamed or removed; new ones can be added."""
    tasks = load_tasks(root)
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        global_notice("error", "That task no longer exists.")
        return

    def field(name: str):
        return ss.get(f"edit::{task_id}::{name}")

    updated, error, n_new = apply_task_edit(
        task, tasks, name=field("name"), rubric=field("rubric"), add_labels=field("add_labels"),
        definitions_str=field("defs"), label_col=field("label_col"), pred_col=field("pred_col"),
        n_candidates=field("ncand"))
    if updated is None:
        global_notice("warning", error)
        return
    save_tasks(root, [updated if t.id == task_id else t for t in tasks])
    ss.pop("_pred_cache", None)
    ss[f"edit::{task_id}::add_labels"] = ""
    flash(f"Saved {updated.name}" + (f" with {n_new} new label(s)" if n_new else ""))


def tasks_tab(app) -> None:
    """Set up, edit and configure labeling tasks."""
    root, tasks = app.root, app.tasks
    with st.container(border=True, key="tasks_panel"):
        st.markdown("##### :material/tune: Tasks")
        for t in tasks:
            title = f"{TYPE_ICONS[t.type]} {plain(t.name)} · {t.type.value}" + (" · archived" if t.archived else "")
            with st.expander(title):
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    for label in t.labels:
                        st.badge(plain(label), color=W.color_for(t.labels, label))
                    st.button("Restore" if t.archived else "Archive",
                              icon=":material/unarchive:" if t.archived else ":material/archive:",
                              key=f"arch::{t.id}", on_click=toggle_archive, args=(root, t.id), type="tertiary",
                              help="Archived tasks are hidden from labeling; their labels are kept.")
                lines = "\n".join(f"{x}: {t.label_definitions.get(x, '')}" for x in t.labels)
                with st.form(f"edit_task::{t.id}", border=False):
                    st.text_input("Name", value=t.name, key=f"edit::{t.id}::name")
                    st.text_area("Rubric (labeling instructions)", value=t.rubric, height=90,
                                 key=f"edit::{t.id}::rubric")
                    st.text_area("Label definitions", value=lines, height=34 * max(len(t.labels), 3) + 20,
                                 key=f"edit::{t.id}::defs", help="One line per label as `label: definition`.")
                    st.text_input("Add labels, comma separated", key=f"edit::{t.id}::add_labels",
                                  help="New labels also need a line in Label definitions above.")
                    if t.type is TaskType.CONTRASTIVE:
                        st.number_input("Candidates per anchor", 2, 10, int(t.n_candidates),
                                        key=f"edit::{t.id}::ncand", help="Applies to samples not yet labeled.")
                    with st.container(horizontal=True):
                        st.text_input("Label column", value=t.label_col, placeholder=f"{t.id}_labels",
                                      key=f"edit::{t.id}::label_col")
                        st.text_input("Prediction column", value=t.pred_col, placeholder=f"{t.id}_pred",
                                      key=f"edit::{t.id}::pred_col")
                    st.form_submit_button("Save changes", icon=":material/save:", on_click=save_task_edit,
                                          args=(root, t.id))
    with st.container(border=True, key="new_task_panel"):
        st.markdown("##### :material/add_circle: New task")
        with st.form("add_task", clear_on_submit=True, border=False):
            st.text_input("Name", key="new_name")
            st.selectbox("Type", [x.value for x in TaskType], key="new_type",
                         format_func=lambda v: TaskType(v).display)
            st.text_input("Labels, comma separated", key="new_labels",
                          help="Token tasks get O automatically. Contrastive defaults to similar, dissimilar.")
            st.checkbox("Multi-label (classification only)", key="new_multi")
            st.number_input("Candidates per anchor (contrastive)", 2, 10, 4, key="new_ncand")
            st.text_area("Rubric (labeling instructions)", key="new_rubric", height=90,
                         help="Shown to annotators beside the labeling controls.")
            st.text_area("Label definitions", key="new_defs", height=110,
                         placeholder="positive: The customer is satisfied.\nnegative: The customer is unhappy.",
                         help="One line per label as `label: definition`. Every label needs one. "
                              "O, similar and dissimilar have built-in defaults.")
            with st.container(horizontal=True):
                st.text_input("Label column (optional)", key="new_label_col",
                              help="Default: {task}_labels, where {task} is the task id (the name as a slug).")
                st.text_input("Prediction column (optional)", key="new_pred_col", help="Default: {task}_pred.")
            st.form_submit_button("Add task", icon=":material/add:", on_click=add_task, args=(root,))
