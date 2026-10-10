"""Sidebar sections: dataset, session, model, queue, export and persistence."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import uncertainty as unc
from dataset import DatasetError, save_upload
from export import build_frame, export_current_jsonl
from label_store import LabelStore
from models import available_models
from paths import DATASETS_DIR, ROOT
from project_config import load_manifest, save_manifest
from review_queue import RANK_MAX, RANK_MEAN, task_predictions
from session import flash, global_notice, ss
from task_defs import Mode
from util import now_iso
from widgets import plain

MODE_LABELS = {"Initial labeling": Mode.INITIAL, "Relabel": Mode.RELABEL}


def import_upload() -> None:
    file = ss.get("upload")
    if file is None:
        return
    try:
        path = save_upload(DATASETS_DIR, file.name, file.getvalue())
        ss["dataset_name"] = os.path.basename(path)
        flash(f"Imported {os.path.basename(path)}")
    except DatasetError as error:
        global_notice("error", f"Could not import {file.name}: {error}")


def retrain(root: str) -> None:
    manifest = load_manifest(root)
    version_ = int(manifest.get("model_version", 0)) + 1
    manifest["model_version"] = version_
    manifest.setdefault("model_history", []).append({"version": version_, "ts": now_iso()})
    save_manifest(root, manifest)
    ss.pop("_pred_cache", None)
    flash(f"Model updated to v{version_}")


def snapshot(store: LabelStore) -> None:
    store.snapshot(force=True)
    flash("Snapshot written")


def data_section(datasets):
    with st.expander("Data", expanded=True, icon=":material/database:"):
        st.file_uploader("Import a dataset", type=["csv", "jsonl"], key="upload",
                         on_change=import_upload, label_visibility="collapsed")
        st.caption("CSV or JSONL with a `text` column and an optional `id`.")
        if not datasets:
            st.caption("No datasets yet.")
            return None
        if ss.get("dataset_name") not in datasets:
            ss.pop("dataset_name", None)
        return st.selectbox("Dataset", datasets, key="dataset_name")


def session_section():
    with st.expander("Session", expanded=True, icon=":material/person:"):
        st.text_input("Annotator", key="annotator", help="Recorded on every label event.")
        ss.setdefault("mode", "Initial labeling")
        choice = st.segmented_control("Mode", list(MODE_LABELS), key="mode", width="stretch")
        st.caption("Initial: work through the dataset. Relabel: review the most uncertain samples.")
    return (ss.get("annotator") or "").strip(), MODE_LABELS.get(choice or "Initial labeling", Mode.INITIAL)


def model_section(root: str, manifest: dict):
    with st.expander("Model", icon=":material/model_training:"):
        model_key = st.selectbox("Backend", available_models(), key="model_key")
        model_version = int(manifest.get("model_version", 0))
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"v{model_version}", color="gray")
            st.button("Retrain", icon=":material/refresh:", key="retrain", on_click=retrain, args=(root,),
                      help="Mock model: re-rolls the random predictions.")
        prefill = st.toggle("Pre-fill from model", key="prefill",
                            help="Pre-annotation speeds labeling but can anchor annotators.")
    return model_key, model_version, bool(prefill)


def queue_section(active, mode: Mode) -> dict:
    with st.expander("Queue", expanded=True, icon=":material/filter_list:"):
        if mode is Mode.INITIAL:
            ss.setdefault("init_order", "Dataset order")
            ss.setdefault("hide_complete", True)
            order = st.segmented_control("Order", ["Dataset order", "Shuffled"], key="init_order", width="stretch")
            hide = st.toggle("Hide complete samples", key="hide_complete",
                             help="A sample is complete once every active task has a saved label.")
            return {"order": order or "Dataset order", "hide": bool(hide)}
        has_disagreement = any(unc.DISAGREE in unc.human_strategies(t) for t in active)
        human = [unc.LOW_CONF] + ([unc.DISAGREE] if has_disagreement else [])
        strategy = st.selectbox("Strategy", list(unc.MODEL_STRATEGIES) + human, key="strategy")
        by_name = {t.name: t.id for t in active}
        rank = st.selectbox("Rank samples by", [RANK_MAX, RANK_MEAN, *by_name], key="rank_by",
                            help="Score each sample from its per-task scores, or from one task only.")
        if strategy in human:
            scope = "Labeled"
            st.caption("Human-signal strategy, so only saved labels are scored.")
        else:
            ss.setdefault("scope", "All")
            scope = st.segmented_control("Scope", ["Labeled", "Unlabeled", "All"], key="scope", width="stretch",
                                         help="Which sample-task pairs count toward a sample's score.") or "All"
        size = st.slider("Queue size", 5, 200, 25, 5, key="queue_size")
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05, key="min_score")
        return {"strategy": strategy, "rank": by_name.get(rank, rank), "scope": scope,
                "size": size, "min_score": min_score}


def persistence_section(store: LabelStore) -> None:
    with st.expander("Persistence", icon=":material/save:"):
        last = (store.last_write_ts or "-")[:19].replace("T", " ")
        snap = store.last_snapshot_seq if store.last_snapshot_seq is not None else "-"
        st.markdown(
            f"Log events **{store.n_events}** · seq {store.seq}  \n"
            f"Current labels **{store.n_current}**  \n"
            f"Last write {plain(last)}  \n"
            f"Last snapshot seq {snap}, every {store.snapshot_every} events"
        )
        st.button("Snapshot now", icon=":material/photo_camera:", key="snapshot",
                  on_click=snapshot, args=(store,))


def export_section(app) -> None:
    store, ds = app.store, app.ds
    stem = os.path.splitext(ds.name)[0]
    wal = Path(store.wal_path).read_bytes() if os.path.exists(store.wal_path) else b""
    with st.expander("Export", icon=":material/download:"):
        st.download_button("Labels JSONL", export_current_jsonl(store, ds.samples, app.tasks),
                           file_name=f"{stem}_labels_seq{store.seq}.jsonl", icon=":material/download:",
                           key="dl_jsonl", disabled=store.n_current == 0)
        with_preds = st.toggle("Include {task}_pred", key="export_preds",
                               help="Adds each task's model probabilities from the loaded backend.")
        try:
            preds = {t.id: task_predictions(app, t) for t in app.tasks} if with_preds else None
            frame = build_frame(store, ds, app.tasks, preds).to_csv(index=False)
        except ValueError as error:
            frame = ""
            st.error(f"Column conflict: {error}", icon=":material/error:")
        st.download_button("Dataset CSV ({task}_labels)", frame,
                           file_name=f"{stem}_columns_seq{store.seq}.csv", icon=":material/table:",
                           key="dl_csv", disabled=not frame)
        st.download_button("Audit log", wal, file_name=f"{stem}_labels.wal.jsonl",
                           icon=":material/receipt_long:", key="dl_wal", disabled=not wal)
        st.caption(f"{plain(ds.name)} · sha256 {ds.sha256[:12]}. Every save is checked and fsynced to "
                   f"{plain(os.path.relpath(store.wal_path, ROOT))} before it is confirmed.")
