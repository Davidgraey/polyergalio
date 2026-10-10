"""Data tab: dataset summary, task definitions and ingestion, then ranking / records / integrity / audit."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

import uncertainty as unc
import widgets as W
from columns import column_conflicts, definitions_frame
from ingest import import_prelabels, pred_column_report
from label_events import summarize_value
from label_store import LabelStore
from review_queue import score_frame
from session import flash, guarded, ss
from widgets import plain


def import_prelabels_cb(store: LabelStore, ds, tasks, annotator: str) -> None:
    report = import_prelabels(store, ds, tasks, annotator)
    ss["_import_report"] = report
    ss.pop("_pred_cache", None)
    flash(f"Imported {sum(r['imported'] for r in report.values())} label(s) from dataset columns")


def data_tab(app, active, labeled, opts, complete) -> None:
    ds, store = app.ds, app.store
    lengths = pd.Series([len(s.text) for s in ds.samples])
    with st.container(border=True, key="data_panel"):
        st.markdown("##### :material/database: Dataset")
        with st.container(horizontal=True, gap="small"):
            st.badge(plain(ds.name), color="gray", icon=":material/description:")
            st.badge(f"{len(ds.samples)} samples", color="blue")
            st.badge(f"{int(lengths.median())} chars median · {int(lengths.max())} max", color="gray")
            st.badge(f"sha256 {ds.sha256[:12]}", color="gray", icon=":material/fingerprint:")
            st.badge(f"{store.n_current} labels · seq {store.seq}", color="green", icon=":material/sell:")
        rows = []
        for t in active:
            recs = list(labeled[t.id].values())
            rows.append({"task": t.name, "type": t.type.value, "labeled": len(recs),
                         "coverage": len(recs) / max(len(ds.samples), 1),
                         "mean confidence": float(np.mean([r["confidence"] for r in recs])) if recs else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"coverage": st.column_config.ProgressColumn("coverage", min_value=0.0,
                                                                                 max_value=1.0, format="percent")})
    guarded(task_definitions, app)
    bottom_sections(app, active, labeled, opts)


def task_definitions(app) -> None:
    """Task contract: type, labels and the {task}_labels / {task}_pred columns, plus dataset ingestion."""
    ds, tasks = app.ds, app.tasks
    with st.container(border=True, key="defs_panel"):
        st.markdown("##### :material/schema: Task definitions")
        st.caption("Per task: `{task}_labels` holds the label as `{\"value\", \"confidence\"}`; `{task}_pred` holds "
                   "the model's raw output. `{task}` is the task id.")
        for problem in column_conflicts(tasks):
            st.error(problem, icon=":material/error:")
        st.dataframe(definitions_frame(tasks, ds), hide_index=True, width="stretch")

        importable = sum(1 for t in tasks for s in ds.samples
                         if t.labels_column.lower() in s.extra and app.store.current(s.id, t.id) is None)
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.button(f"Import {importable} pre-label(s) from dataset", icon=":material/upload:",
                      key="import_prelabels", on_click=import_prelabels_cb,
                      args=(app.store, ds, tasks, app.annotator), disabled=importable == 0 or not app.annotator,
                      help="Validates each {task}_labels cell and writes it to the label log as annotator "
                           "import:<file>. Samples that already have a label are skipped.")
            if importable == 0:
                st.caption("No unlabeled `{task}_labels` cells found in the dataset.")
        report = ss.get("_import_report")
        if report:
            names = {t.id: t.name for t in tasks}
            rows = [{"task": names.get(k, k), "imported": r["imported"], "already labeled": r["existing"],
                     "rejected": len(r["invalid"])} for k, r in report.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            bad = [{"task": names.get(k, k), "sample_id": sid, "problem": why}
                   for k, r in report.items() for sid, why in r["invalid"]]
            if bad:
                st.warning(f"{len(bad)} cell(s) were rejected and not imported.", icon=":material/warning:")
                st.dataframe(pd.DataFrame(bad), hide_index=True, width="stretch")

        preds = pred_column_report(ds, tasks)
        if any(r["ok"] or r["bad"] for r in preds.values()):
            st.caption("Prediction columns found. Choose the **precomputed** backend in the sidebar to use them; "
                       "missing or unreadable cells count as maximally uncertain.")
            names = {t.id: t.name for t in tasks}
            st.dataframe(pd.DataFrame([{"task": names[k], "readable": r["ok"], "unreadable": len(r["bad"]),
                                        "missing": r["missing"]} for k, r in preds.items()]),
                         hide_index=True, width="stretch")
            bad = [{"task": names[k], "sample_id": sid, "problem": why} for k, r in preds.items()
                   for sid, why in r["bad"][:20]]
            if bad:
                st.dataframe(pd.DataFrame(bad), hide_index=True, width="stretch")


def bottom_sections(app, active, labeled, opts) -> None:
    ds, store = app.ds, app.store
    ranking, records, integrity, audit = st.tabs(["Uncertainty ranking", "All records", "Integrity", "Audit log"])
    with ranking:
        df = score_frame(app, active)
        options = list(unc.MODEL_STRATEGIES) + [unc.LOW_CONF, unc.DISAGREE]
        strategy = opts.get("strategy") or st.selectbox("Rank by", options, key="rank_strategy")
        df[strategy] = pd.to_numeric(df[strategy], errors="coerce")
        names = Counter(t.name for t in active)
        columns = {t.id: t.name if names[t.name] == 1 else f"{t.name} ({t.id})" for t in active}
        wide = (df.pivot(index="sample_id", columns="task_id", values=strategy)
                .reindex(index=[s.id for s in ds.samples], columns=[t.id for t in active])
                .rename(columns=columns))
        wide.insert(0, "max", wide.max(axis=1))
        wide.insert(1, "mean", wide.drop(columns=["max"]).mean(axis=1))
        st.caption(f"Per-task {plain(strategy)} scores from {plain(app.model_key)} v{app.model_version}. Higher means "
                   "more uncertain; a sample's queue score is the max or mean across its tasks.")
        values = wide["max"].dropna().astype(float)
        if len(values):
            counts, edges = np.histogram(values, bins=10, range=(0.0, 1.0))
            st.bar_chart(pd.DataFrame({"name": [f"{edges[i]:.1f}" for i in range(10)], "samples": counts}),
                         x="name", y="samples", sort=False, height=W.CHART_HEIGHT)
        st.dataframe(wide.sort_values("max", ascending=False, na_position="last").reset_index(),
                     hide_index=True, width="stretch")

    with records:
        rows = []
        for s in ds.samples:
            row = {"id": s.id, "complete": all(s.id in labeled[t.id] for t in active)}
            for t in active:
                rec = labeled[t.id].get(s.id)
                row[t.name] = summarize_value(t.type.value, rec["value"]) if rec else None
                row[f"{t.name} conf"] = rec["confidence"] if rec else None
            row["text"] = s.text
            rows.append(row)
        st.dataframe(pd.DataFrame(rows, dtype=object), hide_index=True, width="stretch")

    with integrity:
        if app.issues:
            st.dataframe(pd.DataFrame(app.issues), hide_index=True, width="stretch")
        else:
            st.success(f"All {store.n_current} saved label(s) validate against the dataset and task "
                       "definitions.", icon=":material/verified:")

    with audit:
        names = {t.id: t.name for t in app.tasks}
        events = store.events(300)
        if not events:
            st.caption("No label events yet.")
        else:
            st.dataframe(pd.DataFrame([{
                "seq": e["seq"], "time": e["ts"][:19].replace("T", " "), "op": e["op"],
                "annotator": e["annotator"], "mode": e["mode"], "sample_id": e["sample_id"],
                "task": names.get(e["task_id"], e["task_id"]), "confidence": e["confidence"],
                "value": summarize_value(e["task_type"], e["value"]), "model_score": e.get("model_score"),
            } for e in events]), hide_index=True, width="stretch")
