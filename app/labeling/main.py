"""Human-in-the-loop text labeling app.

Run from this folder so ``.streamlit/config.toml`` is picked up::

    streamlit run main.py

Labeling is sample-centric: one current sample is shared by every task, with
one tab per task (classification, token, span, contrastive) editing it. Two
modes: initial labeling (walk the dataset) and relabel (a queue ranked by
uncertainty or human signals).

Saves are validated, then appended to the checksummed, fsynced write-ahead log
in ``label_store.py``. Nothing is overwritten: relabels supersede and removals
retract.

Data lives in ``data/`` next to this file, which ships with a small sample dataset.
Set ``HITL_DATA_DIR`` to use another folder.
"""

import getpass
import os
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_context import AppContext
from dataset import DatasetError, list_datasets
from integrity import integrity_check
from models import create_model
from paths import DATASETS_DIR, PROJECTS_DIR
from project_config import load_manifest, load_tasks, save_manifest
from resources import cached_dataset, get_store
from session import bump_version, guarded, notify, show_notice, ss
from ui.banners import show_store_banners
from ui.page import render_page
from ui.sidebar import (
    data_section,
    export_section,
    model_section,
    persistence_section,
    queue_section,
    session_section,
)
from ui.tasks_tab import tasks_tab
from ui.theme import COMPACT_CSS
from util import now_iso
from widgets import TYPE_ICONS, plain


def tab_labels_for(active) -> dict:
    labels, seen = {}, Counter()
    for t in active:
        label = f"{TYPE_ICONS[t.type]} {t.name}"
        seen[label] += 1
        labels[t.id] = label if seen[label] == 1 else f"{label} ({seen[label]})"
    return labels


def main() -> None:
    st.set_page_config(page_title="HITL labeling", page_icon=":material/label:", layout="wide")
    if "_flash" in ss:
        st.toast(ss.pop("_flash"), icon=":material/check_circle:")
    if "annotator" not in ss:
        try:
            ss["annotator"] = getpass.getuser()
        except Exception:
            ss["annotator"] = ""

    st.html(COMPACT_CSS)
    st.title("HITL labeling", icon=":material/label:")
    show_notice("_notice")

    with st.sidebar:
        ds_name = guarded(data_section, list_datasets(DATASETS_DIR))
    if not ds_name:
        st.caption("Import a CSV or JSONL dataset with a text column from the sidebar.")
        return
    path = os.path.join(DATASETS_DIR, ds_name)
    try:
        stat = os.stat(path)
        ds = cached_dataset(path, stat.st_mtime_ns, stat.st_size)
    except (DatasetError, OSError) as error:
        st.error(f"Could not load {plain(ds_name)}: {error}", icon=":material/error:")
        return

    root = os.path.join(PROJECTS_DIR, ds.project_id)
    store = get_store(root)
    manifest = load_manifest(root)
    if not manifest:
        manifest = {"dataset": ds.name, "sha256": ds.sha256, "n_samples": len(ds.samples),
                    "created": now_iso(), "model_version": 0}
        save_manifest(root, manifest)
    tasks = load_tasks(root)
    active = [t for t in tasks if not t.archived]

    with st.sidebar:
        annotator, mode = session_section()
        model_key, model_version, prefill = model_section(root, manifest)
        opts = (guarded(queue_section, active, mode) or {}) if active else {}

    show_store_banners(store, ds, manifest, root)

    app = AppContext(
        ds=ds, store=store, root=root, tasks=tasks, annotator=annotator, mode=mode,
        model_key=model_key, model_version=model_version, prefill=prefill,
        model=create_model(model_key, version=model_version),
        issues=integrity_check(store, ds.by_id, {t.id: t for t in tasks}),
        pred_cache=ss.setdefault("_pred_cache", {}), notify=notify, bump=bump_version,
    )

    if not active:
        st.info("All tasks are archived. Restore or add one below.", icon=":material/info:")
        guarded(tasks_tab, app)
    else:
        guarded(render_page, app, active, tab_labels_for(active), opts)

    with st.sidebar:
        guarded(export_section, app)
        guarded(persistence_section, store)


if __name__ == "__main__":
    main()
