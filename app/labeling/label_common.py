"""
Shared state, integrity checks, serialization and layout for the labeling tabs.

Each tab keeps one workspace dict in st.session_state under its name. The
workspace is the only source of truth: widgets are keyed with the workspace
version and seeded from it, and their callbacks write back through
`transact`, which edits a deep copy and keeps it only if every label still
validates. Exports are parsed back and compared with the workspace before
they are offered or written.
"""

import copy
import datetime
import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

UPLOAD_TYPES = ["csv", "json", "jsonl"]
CONTEXT_HEIGHT = 240
PREVIEW_HEIGHT = 200
CHART_HEIGHT = 150
MAX_SEGMENTED = 7
HISTORY_SIZE = 5
COLORS = ["blue", "green", "orange", "violet", "red", "yellow", "gray"]
WORD = re.compile(r"\w+(?:['’./\-]\w+)*|[^\w\s]")
PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")


class IntegrityError(ValueError):
    """Labels, records or an export that failed validation."""


# -------------    text helpers    -------------------------------
def split_words(text: str) -> list[str]:
    """Words, numbers and single punctuation marks, in order."""
    return WORD.findall(text or "")


def plain_text(value) -> str:
    """A value as text with Markdown punctuation escaped."""
    text = "-" if value is None else value if isinstance(value, str) else json.dumps(to_json_safe(value))
    return PUNCTUATION.sub(r"\\\1", text)


def lines(text: str) -> list[str]:
    return list(dict.fromkeys(line.strip() for line in (text or "").splitlines() if line.strip()))


def tag_color(tags: list[str], tag: str) -> str:
    return COLORS[tags.index(tag) % len(COLORS)] if tag in tags else "gray"


def badge(text: str, tag: str, color: str) -> str:
    return f":{color}-badge[{plain_text(text)} · {plain_text(tag)}]"


# -------------    serialization    ------------------------------
def to_json_safe(value):
    """
    Plain JSON types for any value a record can hold.

    NaN and infinities become None, numpy scalars and arrays become Python
    numbers and lists, dates become ISO strings, tuples and sets become
    lists, and dict keys become strings.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [to_json_safe(item) for item in items]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def dumps(value) -> str:
    """One strict JSON line; refuses NaN and anything that does not round-trip."""
    safe = to_json_safe(value)
    text = json.dumps(safe, allow_nan=False, ensure_ascii=False)
    if "\n" in text or json.loads(text) != safe:
        raise IntegrityError("value does not survive a JSON round trip")
    return text


def read_records(name: str, data: bytes) -> list[dict]:
    """Rows of a CSV, JSON list or JSONL file as JSON-safe dictionaries."""
    text = data.decode("utf-8-sig")
    if name.endswith(".jsonl"):
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise IntegrityError(f"line {number} is not valid JSON: {error}") from error
    elif name.endswith(".json"):
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
    else:
        frame = pd.read_csv(io.StringIO(text), dtype=object, keep_default_na=False, na_values=[""])
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    if not isinstance(rows, list) or not rows:
        raise IntegrityError("the file holds no records")
    bad = [number for number, row in enumerate(rows, 1) if not isinstance(row, dict)]
    if bad:
        raise IntegrityError(f"records {bad[:10]} are not objects")
    return [to_json_safe(row) for row in rows]


def record_columns(records: list[dict]) -> list[str]:
    return list(dict.fromkeys(key for record in records for key in record))


def text_column(records: list[dict]) -> str | None:
    """"text" if present, else the first column holding strings."""
    columns = record_columns(records)
    if "text" in columns:
        return "text"
    return next((c for c in columns if any(isinstance(r.get(c), str) for r in records)), None)


# -------------    workspace    ----------------------------------
def blank_workspace(config: dict) -> dict:
    return dict(
        records=[], ids=[], labels={}, drafts={}, position=0, source=None,
        config=copy.deepcopy(config), view={}, notice=None, version=0, config_version=0,
    )


def workspace(task) -> dict:
    """The task's workspace, created with its default config on first use."""
    return st.session_state.setdefault(task.NAME, blank_workspace(task.DEFAULT_CONFIG))


def widget_key(space: dict, name: str, base: str) -> str:
    """Key for a record-bound widget; a new version gives a fresh widget seeded from the workspace."""
    return f"{name}_{base}_{space['version']}"


def config_key(space: dict, name: str, field: str) -> str:
    return f"{name}_cfg_{field}_{space['config_version']}"


def check_workspace(task, space: dict) -> list[str]:
    """Every problem with the workspace's structure and saved labels."""
    problems = []
    records, ids, labels = space["records"], space["ids"], space["labels"]
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        return ["records are not a list of objects"]
    if len(ids) != len(records):
        problems.append(f"{len(ids)} ids for {len(records)} records")
    if len(set(map(str, ids))) != len(ids):
        problems.append("record ids are not unique")
    if records and not 0 <= space["position"] < len(records):
        problems.append(f"position {space['position']} is outside the records")
    for index, label in labels.items():
        if not isinstance(index, int) or not 0 <= index < len(records):
            problems.append(f"label key {index!r} is not a record index")
            continue
        for problem in task.validate_label(label):
            problems.append(f"row {index + 1}: {problem}")
        if label.get("record") != records[index]:
            problems.append(f"row {index + 1}: saved sample differs from the record")
        try:
            dumps(label)
        except (IntegrityError, TypeError, ValueError) as error:
            problems.append(f"row {index + 1}: {error}")
    return problems


def set_notice(space: dict, level: str, text: str) -> None:
    space["notice"] = {"level": level, "text": text}


def transact(task, action, *args) -> bool:
    """
    Apply action(space, *args) to a deep copy of the workspace and keep the
    copy only if it passes check_workspace; otherwise the workspace is left
    untouched and the reason is shown.
    """
    space = workspace(task)
    trial = copy.deepcopy(space)
    trial["notice"] = None
    try:
        action(trial, *args)
        problems = check_workspace(task, trial)
    except Exception as error:
        problems = [f"{type(error).__name__}: {error}"]
    if problems:
        set_notice(space, "error", "Change rejected, nothing was modified. " + "; ".join(problems[:5]))
        return False
    space.clear()
    space.update(trial)
    return True


def bump(space: dict) -> None:
    space["version"] += 1


def remember(task) -> None:
    """Keep a copy of the workspace before it is replaced."""
    space = workspace(task)
    if space["records"]:
        history = st.session_state.setdefault(f"{task.NAME}_history", [])
        history.append(copy.deepcopy(space))
        del history[:-HISTORY_SIZE]


def restore(task) -> None:
    history = st.session_state.get(f"{task.NAME}_history") or []
    if history:
        previous = history.pop()
        previous["version"] += 1
        previous["config_version"] += 1
        st.session_state[task.NAME] = previous


def replace(task, records: list[dict], ids: list, labels: dict, config: dict, source) -> None:
    """Load a new dataset, keeping the old one in the history; autosave is switched off so the old file is never overwritten."""
    remember(task)
    space = workspace(task)
    fresh = blank_workspace({**config, "autosave": ""})
    fresh.update(
        records=copy.deepcopy(records), ids=list(ids), labels=copy.deepcopy(labels), source=source,
        version=space["version"] + 1, config_version=space["config_version"] + 1,
    )
    problems = check_workspace(task, fresh)
    if problems:
        raise IntegrityError("; ".join(problems[:5]))
    st.session_state[task.NAME] = fresh


# -------------    export and import    --------------------------
def export_text(task, space: dict) -> str:
    """
    JSONL of the saved labels, one line per labeled record in row order.

    The text is parsed back and every line must rebuild exactly the stored
    record and label, otherwise IntegrityError is raised.
    """
    problems = check_workspace(task, space)
    if problems:
        raise IntegrityError("; ".join(problems[:5]))
    order = sorted(space["labels"])
    rows = [{"id": space["ids"][index], **task.to_row(space["labels"][index])} for index in order]
    text = "".join(dumps(row) + "\n" for row in rows)
    parsed = [json.loads(line) for line in text.splitlines()]
    if len(parsed) != len(order):
        raise IntegrityError(f"export has {len(parsed)} lines for {len(order)} labels")
    for index, row in zip(order, parsed):
        rebuilt = task.from_row(row)
        if to_json_safe(rebuilt) != to_json_safe(space["labels"][index]):
            raise IntegrityError(f"row {index + 1} does not rebuild from its exported line")
        if row["id"] != to_json_safe(space["ids"][index]):
            raise IntegrityError(f"row {index + 1} exported the wrong id")
    return text


def import_labeled(task, rows: list[dict]) -> tuple[list[dict], list, dict]:
    """Records, ids and labels from an exported JSONL; all rows must be valid."""
    records, ids, labels, problems = [], [], {}, []
    for number, row in enumerate(rows, 1):
        try:
            label = task.from_row(row)
            issues = task.validate_label(label)
        except Exception as error:
            issues = [f"{type(error).__name__}: {error}"]
        if issues:
            problems.append(f"line {number}: {'; '.join(issues)}")
            continue
        labels[len(records)] = label
        records.append(label["record"])
        ids.append(row.get("id", number - 1))
    if len(set(map(str, ids))) != len(ids):
        problems.append("ids are not unique")
    if problems:
        raise IntegrityError("File rejected, nothing was loaded. " + " | ".join(problems[:5]))
    return records, ids, labels


def write_atomic(path: str, text: str) -> Path:
    """Write text through a temporary file, keep a .bak of the old file, and verify the result."""
    target = Path(path).expanduser().resolve()
    if target.suffix != ".jsonl":
        raise IntegrityError("autosave path must end in .jsonl")
    if not target.parent.is_dir():
        raise IntegrityError(f"folder {target.parent} does not exist")
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.read_text(encoding="utf-8") != text:
        raise IntegrityError("temporary file did not read back identically")
    if target.exists():
        os.replace(target, target.with_name(target.name + ".bak"))
    os.replace(temporary, target)
    if target.read_text(encoding="utf-8") != text:
        raise IntegrityError("autosave file did not read back identically")
    return target


def autosave(task) -> None:
    """Write the verified export to the autosave path, if one is set."""
    space = workspace(task)
    path = space["config"].get("autosave", "").strip()
    if not path:
        return
    try:
        target = write_atomic(path, export_text(task, space))
        space["view"]["saved_to"] = str(target)
    except Exception as error:
        set_notice(space, "error", f"Autosave failed, labels are kept in this session: {error}")


def commit(task, action, *args) -> None:
    """transact, then autosave when the change was kept."""
    if transact(task, action, *args):
        autosave(task)


# -------------    navigation    ---------------------------------
def move_to(space: dict, position: int) -> None:
    if space["records"]:
        space["position"] = min(max(int(position), 0), len(space["records"]) - 1)
    space["view"] = {key: value for key, value in space["view"].items() if key == "saved_to"}
    bump(space)


def step(space: dict, offset: int) -> None:
    move_to(space, space["position"] + offset)


def next_unlabeled(space: dict) -> None:
    pending = [index for index in range(len(space["records"])) if index not in space["labels"]]
    later = [index for index in pending if index > space["position"]]
    move_to(space, (later or pending or [space["position"]])[0])


def navigation(task) -> None:
    space = workspace(task)
    name = task.NAME
    with st.container(horizontal=True):
        st.button(
            "Previous", icon=":material/arrow_back:", key=f"{name}_previous",
            on_click=transact, args=(task, step, -1), disabled=space["position"] == 0,
        )
        st.button(
            "Next", icon=":material/arrow_forward:", key=f"{name}_next",
            on_click=transact, args=(task, step, 1), disabled=space["position"] >= len(space["records"]) - 1,
        )
        st.button(
            "Next unlabeled", icon=":material/skip_next:", key=f"{name}_unlabeled",
            on_click=transact, args=(task, next_unlabeled),
        )


# -------------    sidebar pieces    -----------------------------
def upload_file(task, key: str) -> None:
    file = st.session_state.get(key)
    if file is None:
        return
    space = workspace(task)
    try:
        rows = read_records(file.name, file.getvalue())
        if all(task.is_labeled_row(row) for row in rows):
            records, ids, labels = import_labeled(task, rows)
            config = task.config_from_labels(labels, space["config"])
        else:
            records, ids, labels = rows, list(range(len(rows))), {}
            config = task.config_for_records(records, space["config"])
        replace(task, records, ids, labels, config, ("file", file.name))
    except Exception as error:
        set_notice(space, "error", f"Could not load {file.name}: {error}")


def load_example(task) -> None:
    name = st.session_state[f"{task.NAME}_example"]
    try:
        records = to_json_safe(task.EXAMPLES[name]["records"])
        config = task.config_for_example(task.EXAMPLES[name], records, workspace(task)["config"])
        replace(task, records, list(range(len(records))), {}, config, ("example", name))
    except Exception as error:
        set_notice(workspace(task), "error", f"Could not load {name}: {error}")


def data_section(task) -> None:
    space = workspace(task)
    name = task.NAME
    with st.expander("Data", expanded=True, icon=":material/database:"):
        key = f"{name}_upload_{space['config_version']}"
        st.file_uploader(
            "Records or labeled JSONL", type=UPLOAD_TYPES, key=key, label_visibility="collapsed",
            on_change=upload_file, args=(task, key),
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.selectbox("Example task", list(task.EXAMPLES), key=f"{name}_example", label_visibility="collapsed")
            st.button("Load", icon=":material/download:", key=f"{name}_load", on_click=load_example, args=(task,))
        if st.session_state.get(f"{name}_history"):
            st.button(
                "Restore previous data", icon=":material/undo:", key=f"{name}_restore",
                on_click=restore, args=(task,),
            )


def set_config(task, field: str, key: str, clean) -> None:
    def action(space):
        space["config"][field] = clean(st.session_state[key])
        space["drafts"] = {}
        bump(space)
    transact(task, action)


def config_widget(task, widget, label: str, field: str, clean=lambda value: value, **kwargs):
    """A sidebar widget seeded from and written back to workspace config."""
    space = workspace(task)
    key = config_key(space, task.NAME, field)
    value = space["config"].get(field)
    seed = {"value": value}
    if widget in (st.selectbox, st.radio):
        options = list(kwargs.get("options", []))
        seed = {"index": options.index(value) if value in options else None}
    elif widget is st.multiselect:
        seed = {"default": [item for item in (value or []) if item in kwargs.get("options", [])]}
    return widget(label, key=key, on_change=set_config, args=(task, field, key, clean), **seed, **kwargs)


# -------------    main area pieces    ---------------------------
def toolbar(task) -> str | None:
    """Autosave path and download; returns the verified export or None."""
    space = workspace(task)
    name = task.NAME
    export, failure = None, None
    try:
        export = export_text(task, space)
    except Exception as error:
        failure = str(error)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        config_widget(
            task, st.text_input, "Autosave path", "autosave", clean=str.strip,
            placeholder=f"{name}_labels.jsonl", width=320,
        )
        st.download_button(
            "Download JSONL", export or "", file_name=f"{name}_labels.jsonl", icon=":material/download:",
            key=f"{name}_download", disabled=not export,
        )
    if space["view"].get("saved_to") and space["config"].get("autosave"):
        st.caption(f"Autosaving to {plain_text(space['view']['saved_to'])}")
    if failure:
        st.error(f"Export blocked, labels failed verification: {failure}", icon=":material/error:")
    return export


def notice(space: dict) -> None:
    message = space.get("notice")
    if message:
        show = st.error if message["level"] == "error" else st.warning
        show(message["text"], icon=":material/error:" if message["level"] == "error" else ":material/warning:")


def metrics(space: dict) -> None:
    labeled, size = len(space["labels"]), len(space["records"])
    with st.container(horizontal=True, gap="small"):
        st.metric("Labeled", f"{labeled} / {size}", border=True)
        st.metric("Remaining", size - labeled, border=True)
        st.metric("Progress", f"{100 * labeled / max(size, 1):.0f}%", border=True)
        st.metric("Current row", space["position"] + 1, border=True)


def saved_badge(saved: bool) -> None:
    st.badge("saved" if saved else "unsaved", color="green" if saved else "gray")


def distribution_card(counts: pd.DataFrame, color: str | None = None) -> None:
    """Bar chart of a frame with name and count columns."""
    with st.container(border=True):
        st.subheader("Label distribution")
        if counts.empty or counts["count"].sum() == 0:
            st.caption("No labels yet.")
            return
        st.bar_chart(counts, x="name", y="count", color=color, sort=False, height=CHART_HEIGHT)


def display_value(value) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(to_json_safe(value), ensure_ascii=False)


def records_table(space: dict, summary) -> None:
    """Every record as text columns, so mixed types never break the table."""
    rows = [
        {
            "id": display_value(space["ids"][i]),
            "label": summary(space["labels"][i]) if i in space["labels"] else None,
            **{key: display_value(value) for key, value in record.items() if key not in ("id", "label")},
        }
        for i, record in enumerate(space["records"])
    ]
    with st.expander("All records", icon=":material/table_rows:"):
        st.dataframe(pd.DataFrame(rows, dtype=object), width="stretch", hide_index=True)


def page(task) -> None:
    """Shared page frame: integrity check, toolbar, notice and empty state around task.body."""
    space = workspace(task)
    problems = check_workspace(task, space)
    if problems:
        st.error("Workspace failed its integrity check: " + "; ".join(problems[:5]), icon=":material/error:")
    toolbar(task)
    notice(space)
    if not space["records"]:
        st.caption("Upload records or load an example task from the sidebar.")
        return
    metrics(space)
    task.body(space)
