"""Sequence labeling: typed spans with movable start and end, plus one label for the whole text."""

import sys
from collections import Counter

import pandas as pd
import streamlit as st
from label_common import (
    CONTEXT_HEIGHT,
    MAX_SEGMENTED,
    IntegrityError,
    badge,
    bump,
    commit,
    config_widget,
    data_section,
    distribution_card,
    lines,
    navigation,
    page,
    plain_text,
    record_columns,
    records_table,
    saved_badge,
    split_words,
    step,
    tag_color,
    text_column,
    transact,
    widget_key,
    workspace,
)
from label_tasks import SEQUENCE_TASKS

NAME = "sequence"
TASK = sys.modules[__name__]
EXAMPLES = SEQUENCE_TASKS
DEFAULT_CONFIG = dict(span_types_text="", labels_text="", field=None, autosave="")


def span_types(config: dict) -> list[str]:
    return lines(config["span_types_text"])


def label_set(config: dict) -> list[str]:
    return lines(config["labels_text"])


def record_text(record: dict, field: str | None) -> str:
    value = record.get(field) if field else None
    return value if isinstance(value, str) else "" if value is None else str(value)


def normalize_range(value, size: int) -> tuple[int, int]:
    """Any slider value (int, pair or None) as an ordered in-range (start, end)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = sorted(int(v) for v in value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        start = end = int(value)
    else:
        start = end = 0
    last = max(size - 1, 0)
    return min(max(start, 0), last), min(max(end, 0), last)


def bio_tags(size: int, spans: list[dict]) -> list[str]:
    """B-/I-/O tag per token for non-overlapping inclusive spans."""
    tags = ["O"] * size
    for span in spans:
        tags[span["start"]] = f"B-{span['type']}"
        for index in range(span["start"] + 1, span["end"] + 1):
            tags[index] = f"I-{span['type']}"
    return tags


def span_text(tokens: list[str], span: dict) -> str:
    return " ".join(tokens[span["start"]: span["end"] + 1])


def draft(space: dict) -> dict:
    """Editable spans and label for the current row, from its saved label or empty."""
    position = space["position"]
    saved = space["labels"].get(position)
    current = space["drafts"].get(position)
    if current is None or (saved is None and current["field"] != space["config"]["field"]):
        if saved is not None:
            current = {
                "field": saved["field"], "text": saved["text"], "tokens": list(saved["tokens"]),
                "spans": [dict(span) for span in saved["spans"]], "label": saved["label"],
            }
        else:
            text = record_text(space["records"][position], space["config"]["field"])
            current = {"field": space["config"]["field"], "text": text, "tokens": split_words(text), "spans": [], "label": None}
        space["drafts"][position] = current
    return current


# -------------    label contract    -----------------------------
def span_problems(spans, size: int) -> list[str]:
    if not isinstance(spans, list):
        return ["spans are not a list"]
    problems, previous_end = [], -1
    for number, span in enumerate(spans, 1):
        if not isinstance(span, dict) or set(span) != {"start", "end", "type"}:
            problems.append(f"span {number} must have exactly start, end and type")
            continue
        start, end, kind = span["start"], span["end"], span["type"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
            problems.append(f"span {number} start and end must be integers")
            continue
        if not 0 <= start <= end < size:
            problems.append(f"span {number} ({start}, {end}) is outside {size} tokens")
        if start <= previous_end:
            problems.append(f"span {number} overlaps or is out of order")
        if not isinstance(kind, str) or not kind:
            problems.append(f"span {number} has no type")
        previous_end = max(previous_end, end)
    return problems


def validate_label(label) -> list[str]:
    if not isinstance(label, dict):
        return ["label is not an object"]
    problems = []
    tokens = label.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t for t in tokens):
        problems.append("tokens must be a non-empty list of non-empty strings")
        tokens = []
    problems.extend(span_problems(label.get("spans"), len(tokens)))
    if label.get("label") is not None and not (isinstance(label["label"], str) and label["label"]):
        problems.append("sequence label must be text or empty")
    if not isinstance(label.get("text"), str):
        problems.append("text is missing")
    if not isinstance(label.get("record"), dict):
        problems.append("the sample record is missing")
    elif record_text(label["record"], label.get("field")) != label.get("text"):
        problems.append("text does not match the record field")
    return problems


def to_row(label: dict) -> dict:
    tokens = label["tokens"]
    return {
        "text": label["text"], "field": label["field"], "tokens": tokens,
        "spans": [{**span, "text": span_text(tokens, span)} for span in label["spans"]],
        "bio": bio_tags(len(tokens), label["spans"]), "label": label["label"], "record": label["record"],
    }


def from_row(row: dict) -> dict:
    tokens, raw = row["tokens"], row["spans"]
    if not isinstance(tokens, list) or not isinstance(raw, list):
        raise IntegrityError("tokens and spans must be lists")
    spans = [{"start": s["start"], "end": s["end"], "type": s["type"]} for s in raw]
    problems = span_problems(spans, len(tokens))
    if problems:
        raise IntegrityError("; ".join(problems))
    for number, (span, source) in enumerate(zip(spans, raw), 1):
        if "text" in source and source["text"] != span_text(tokens, span):
            raise IntegrityError(f"span {number} text does not match its tokens")
    if "bio" in row and row["bio"] != bio_tags(len(tokens), spans):
        raise IntegrityError("bio tags do not match the spans")
    text = row.get("text") if isinstance(row.get("text"), str) else " ".join(map(str, tokens))
    field = row.get("field") or "text"
    record = row.get("record", {field: text})
    return {"record": record, "field": field, "text": text, "tokens": tokens, "spans": spans, "label": row.get("label")}


def is_labeled_row(row: dict) -> bool:
    return {"tokens", "spans"} <= row.keys()


def config_for_example(example: dict, records: list[dict], current: dict) -> dict:
    return {
        **current, "span_types_text": "\n".join(example["span_types"]),
        "labels_text": "\n".join(example["labels"]), "field": text_column(records),
    }


def config_for_records(records: list[dict], current: dict) -> dict:
    return {**current, "field": text_column(records)}


def config_from_labels(labels: dict, current: dict) -> dict:
    types = [*span_types(current), *(s["type"] for label in labels.values() for s in label["spans"])]
    names = [*label_set(current), *(label["label"] for label in labels.values() if label["label"])]
    return {
        **current, "span_types_text": "\n".join(dict.fromkeys(types)),
        "labels_text": "\n".join(dict.fromkeys(names)), "field": labels[min(labels)]["field"],
    }


# -------------    actions    ------------------------------------
def selected_span(space: dict) -> dict:
    current = draft(space)
    start, end = normalize_range(space["view"].get("range"), len(current["tokens"]))
    kind = space["view"].get("span_type")
    if kind not in span_types(space["config"]):
        raise IntegrityError("choose a span type")
    return {"start": start, "end": end, "type": kind}


def place_span(spans: list[dict], span: dict, skip: int | None = None) -> list[dict]:
    kept = [s for i, s in enumerate(spans) if i != skip]
    if any(s["start"] <= span["end"] and span["start"] <= s["end"] for s in kept):
        raise IntegrityError("that span overlaps an existing one")
    return sorted([*kept, span], key=lambda s: s["start"])


def set_range(space: dict, key: str) -> None:
    space["view"]["range"] = normalize_range(st.session_state[key], len(draft(space)["tokens"]))


def set_span_type(space: dict, key: str) -> None:
    space["view"]["span_type"] = st.session_state[key]


def pick_span(space: dict, key: str) -> None:
    index = st.session_state[key]
    space["view"]["pick"] = index
    if index is not None:
        span = draft(space)["spans"][index]
        space["view"]["range"] = (span["start"], span["end"])
        space["view"]["span_type"] = span["type"]
    bump(space)


def add_span(space: dict) -> None:
    current = draft(space)
    current["spans"] = place_span(current["spans"], selected_span(space))
    space["view"]["pick"] = None
    bump(space)


def update_span(space: dict) -> None:
    current = draft(space)
    index = space["view"].get("pick")
    if index is None or not 0 <= index < len(current["spans"]):
        raise IntegrityError("pick a span to update")
    current["spans"] = place_span(current["spans"], selected_span(space), skip=index)
    space["view"]["pick"] = None
    bump(space)


def delete_span(space: dict) -> None:
    current = draft(space)
    index = space["view"].get("pick")
    if index is None or not 0 <= index < len(current["spans"]):
        raise IntegrityError("pick a span to delete")
    current["spans"].pop(index)
    space["view"]["pick"] = None
    bump(space)


def set_label(space: dict, key: str) -> None:
    draft(space)["label"] = st.session_state[key]


def save(space: dict) -> None:
    current = draft(space)
    position = space["position"]
    if not current["tokens"]:
        raise IntegrityError("this row has no tokens to save")
    space["labels"][position] = {"record": space["records"][position], **current}
    space["drafts"].pop(position, None)
    step(space, 1)


def discard(space: dict) -> None:
    space["drafts"].pop(space["position"], None)
    space["view"]["pick"] = None
    bump(space)


def unlabel(space: dict) -> None:
    space["labels"].pop(space["position"], None)
    space["drafts"].pop(space["position"], None)
    bump(space)


# -------------    layout    -------------------------------------
def sidebar() -> None:
    space = workspace(TASK)
    data_section(TASK)
    config_widget(TASK, st.selectbox, "Text field", "field", options=record_columns(space["records"]))
    with st.expander("Task", expanded=True, icon=":material/tune:"):
        config_widget(TASK, st.text_area, "Span types, one per line", "span_types_text", height=90)
        config_widget(TASK, st.text_area, "Sequence labels, one per line", "labels_text", height=90)


def highlighted(current: dict, types: list[str]) -> str:
    """Text with each span shown as a colored badge."""
    tokens, parts, cursor = current["tokens"], [], 0
    for span in current["spans"]:
        parts.extend(plain_text(word) for word in tokens[cursor: span["start"]])
        parts.append(badge(span_text(tokens, span), span["type"], tag_color(types, span["type"])))
        cursor = span["end"] + 1
    parts.extend(plain_text(word) for word in tokens[cursor:])
    return " ".join(parts)


def span_card(space: dict, current: dict, types: list[str]) -> None:
    tokens = current["tokens"]
    with st.container(border=True):
        st.subheader("Spans")
        if not types:
            st.caption("Add span types in the sidebar.")
            return
        if not tokens:
            st.caption("This row has no tokens.")
            return
        start, end = normalize_range(space["view"].get("range"), len(tokens))
        if len(tokens) > 1:
            key = widget_key(space, NAME, "range")
            st.select_slider(
                "Start and end", list(range(len(tokens))), value=(start, end), key=key,
                format_func=lambda i: tokens[i], on_change=transact, args=(TASK, set_range, key),
            )
        st.caption(f"Selected: {plain_text(' '.join(tokens[start: end + 1]))}")
        kind = space["view"].get("span_type")
        key = widget_key(space, NAME, "span_type")
        st.segmented_control(
            "Type", types, default=kind if kind in types else None, key=key,
            on_change=transact, args=(TASK, set_span_type, key), label_visibility="collapsed",
        )
        picked = space["view"].get("pick")
        with st.container(horizontal=True):
            st.button("Add span", icon=":material/add:", key="sequence_add", on_click=transact, args=(TASK, add_span))
            st.button("Update", icon=":material/edit:", key="sequence_update", on_click=transact, args=(TASK, update_span), disabled=picked is None)
            st.button("Delete", icon=":material/delete:", key="sequence_delete", on_click=transact, args=(TASK, delete_span), disabled=picked is None)
        if current["spans"]:
            key = widget_key(space, NAME, "pick")
            st.pills(
                "Spans", list(range(len(current["spans"]))), default=picked, key=key,
                on_change=transact, args=(TASK, pick_span, key), label_visibility="collapsed",
                format_func=lambda i: badge(
                    span_text(tokens, current["spans"][i]), current["spans"][i]["type"],
                    tag_color(types, current["spans"][i]["type"]),
                ),
            )
            st.caption("Pick a span to move its start and end, then update it.")


def label_card(space: dict, current: dict, labels: list[str]) -> None:
    saved = space["labels"].get(space["position"])
    changed = saved is not None and any(saved[k] != current[k] for k in ("tokens", "spans", "label"))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Sequence label")
            saved_badge(saved is not None and not changed)
        options = list(dict.fromkeys([*labels, *([current["label"]] if current["label"] else [])]))
        if options:
            key = widget_key(space, NAME, "label")
            settings = dict(key=key, on_change=transact, args=(TASK, set_label, key), label_visibility="collapsed")
            if len(options) > MAX_SEGMENTED:
                st.radio("Label", options, index=options.index(current["label"]) if current["label"] else None, **settings)
            else:
                st.segmented_control("Label", options, default=current["label"], width="stretch", **settings)
        else:
            st.caption("Add sequence labels in the sidebar.")
        with st.container(horizontal=True):
            st.button("Save and next", icon=":material/check:", key="sequence_save", type="primary", on_click=commit, args=(TASK, save), disabled=not current["tokens"])
            st.button("Discard edits", icon=":material/restart_alt:", key="sequence_discard", on_click=transact, args=(TASK, discard))
            st.button("Remove label", icon=":material/delete:", key="sequence_unlabel", on_click=commit, args=(TASK, unlabel), disabled=saved is None)
        navigation(TASK)


def body(space: dict) -> None:
    if not space["config"]["field"]:
        st.caption("Choose the text field in the sidebar.")
        return
    types, labels = span_types(space["config"]), label_set(space["config"])
    if space["view"].get("span_type") is None and types:
        space["view"]["span_type"] = types[0]
    current = draft(space)
    position, size = space["position"], len(space["records"])
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader(f"Text, row {position + 1} of {size}")
            with st.container(height=CONTEXT_HEIGHT):
                st.markdown(highlighted(current, types) or "This row has no text.")
        span_card(space, current, types)
    with right:
        label_card(space, current, labels)
        saved = list(space["labels"].values())
        span_counts = Counter(span["type"] for label in saved for span in label["spans"])
        label_counts = Counter(label["label"] for label in saved if label["label"])
        type_names = list(dict.fromkeys([*types, *span_counts]))
        label_names = list(dict.fromkeys([*labels, *label_counts]))
        distribution_card(
            pd.DataFrame(
                {
                    "name": [*type_names, *label_names],
                    "count": [*(span_counts[t] for t in type_names), *(label_counts[n] for n in label_names)],
                    "kind": ["span"] * len(type_names) + ["label"] * len(label_names),
                }
            ),
            color="kind",
        )
    records_table(space, lambda label: ", ".join(f"{s['type']}: {span_text(label['tokens'], s)}" for s in label["spans"]))


def render() -> None:
    page(TASK)
