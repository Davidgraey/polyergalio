"""Warnings about the label log and the dataset file, shown above the tabs."""

from __future__ import annotations

import os

import streamlit as st

from paths import ROOT
from project_config import load_manifest, save_manifest
from util import now_iso
from widgets import plain


def accept_dataset(root: str, sha: str) -> None:
    manifest = load_manifest(root)
    manifest.setdefault("sha256_history", []).append({"sha256": manifest.get("sha256"), "replaced": now_iso()})
    manifest["sha256"] = sha
    save_manifest(root, manifest)


def show_store_banners(store, ds, manifest: dict, root: str) -> None:
    report = store.report
    if report["quarantined"]:
        st.error(f"{report['quarantined']} corrupt line(s) in the label log were moved to "
                 f"{plain(os.path.relpath(store.quarantine_path, ROOT))}. All other labels loaded normally.",
                 icon=":material/error:")
    if report["truncated_tail"]:
        st.warning("The last write before the previous shutdown was incomplete. It was quarantined and new "
                   "writes are kept separate from it.", icon=":material/warning:")
    if report["recovered_from_snapshot"]:
        st.warning(f"The label log was missing and was rebuilt from {plain(report['recovered_from_snapshot'])}.",
                   icon=":material/warning:")
    if manifest.get("sha256") != ds.sha256:
        st.error("This dataset file changed after labeling started. Saved labels are re-checked under "
                 "Integrity.", icon=":material/error:")
        st.button("Accept current file version", icon=":material/done:", key="accept_ds",
                  on_click=accept_dataset, args=(root, ds.sha256))
    for warning in ds.warnings:
        st.warning(warning, icon=":material/warning:")
