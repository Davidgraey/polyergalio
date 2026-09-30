"""Token labeling: one tag per word, POS or BIO style."""

import sys
from collections import Counter

import pandas as pd
import streamlit as st
from label_common import (
    CONTEXT_HEIGHT,
    PREVIEW_HEIGHT,
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
from label_tasks import TOKEN_TASKS

NAME = "token"
TASK = sys.modules[__name__]
EXAMPLES = TOKEN_TASKS
DEFAULT_CONFIG = dict(tags_text="O", field=None, autosave="")


def tag_set(config: dict) -> list[str]:
    return lines(config["tags_text"]) or ["O"]


def record_text(record: dict, field: str | None) -> str:
    value = record.get(field) if field else None
    return value if isinstance(value, str) else "" if value is None else str(value)


def draft(space: dict) -> dict:
    """Editable tags for the current row, from its saved label or the default tag."""
    position = space["position"]
    saved = space["labels"].get(position)
    current = space["drafts"].get(position)
    if current is None or (saved is None and current["field"] != space["config"]["field"]):
        if saved is not None:
            current = {key: list(saved[key]) if isinstance(saved[key], list) else saved[key] for key in ("field", "text", "tokens", "tags")}
        else:
            text = record_text(space["records"][position], space["config"]["field"])
            tokens = split_words(text)
            current = {"field": space["config"]["field"], "text": text, "tokens": tokens, "tags": [tag_set(space["config"])[0]] * len(tokens)}
        space["drafts"][position] = current
    return current


# -------------    label contract    -----------------------------
def validate_label(label) -> list[str]:
    if not isinstance(label, dict):
        return ["label is not an object"]
    problems = []
    tokens, tags = label.get("tokens"), label.get("tags")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t for t in tokens):
        problems.append("tokens must be a non-empty list of non-empty strings")
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        problems.append("tags must be a list of non-empty strings")
    elif isinstance(tokens, list) and len(tags) != len(tokens):
        problems.append(f"{len(tags)} tags for {len(tokens)} tokens")
    if not isinstance(label.get("text"), str):
        problems.append("text is missing")
    if not isinstance(label.get("record"), dict):
        problems.append("the sample record is missing")
    elif label.get("field") is not None and record_text(label["record"], label["field"]) != label.get("text"):
        problems.append("text does not match the record field")
    return problems


def to_row(label: dict) -> dict:
    return {
        "text": label["text"], "field": label["field"], "tokens": label["tokens"],
        "tags": label["tags"], "record": label["record"],
    }


def from_row(row: dict) -> dict:
    if not isinstance(row.get("tokens"), list):
        raise IntegrityError("tokens are not a list")
    text = row.get("text") if isinstance(row.get("text"), str) else " ".join(map(str, row["tokens"]))
    field = row.get("field")
    record = row.get("record", {field or "text": text})
    return {"record": record, "field": field or "text", "text": text, "tokens": row["tokens"], "tags": row["tags"]}


def is_labeled_row(row: dict) -> bool:
    return {"tokens", "tags"} <= row.keys()


def config_for_example(example: dict, records: list[dict], current: dict) -> dict:
    return {**current, "tags_text": "\n".join(example["tags"]), "field": text_column(records)}


def config_for_records(records: list[dict], current: dict) -> dict:
    return {**current, "field": text_column(records)}


def config_from_labels(labels: dict, current: dict) -> dict:
    tags = list(dict.fromkeys(tag for label in labels.values() for tag in label["tags"]))
    default = tag_set(current)[0]
    ordered = [default, *(tag for tag in tags if tag != default)] if default in tags else tags
    return {**current, "tags_text": "\n".join(ordered), "field": labels[min(labels)]["field"]}


# -------------    actions    ------------------------------------
def set_selection(space: dict, key: str) -> None:
    space["view"]["selection"] = sorted(int(i) for i in st.session_state[key] or [])


def apply_tag(space: dict, tag: str) -> None:
    current = draft(space)
    for index in space["view"].get("selection", []):
        if not 0 <= index < len(current["tags"]):
            raise IntegrityError(f"token {index} is outside the row")
        current["tags"][index] = tag
    space["view"]["selection"] = []
    bump(space)


def select_default(space: dict) -> None:
    current = draft(space)
    default = tag_set(space["config"])[0]
    space["view"]["selection"] = [i for i, tag in enumerate(current["tags"]) if tag == default]
    bump(space)


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
    space["view"]["selection"] = []
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
        config_widget(TASK, st.text_area, "Tags, one per line", "tags_text", height=120)
        st.caption("The first tag is the default for untagged tokens.")


def token_label(word: str, tag: str, tags: list[str]) -> str:
    if tag == tags[0]:
        return plain_text(word)
    return f"{plain_text(word)} :{tag_color(tags, tag)}-badge[{plain_text(tag)}]"


def tagged_text(current: dict, tags: list[str]) -> str:
    return " ".join(
        plain_text(word) if tag == tags[0] else badge(word, tag, tag_color(tags, tag))
        for word, tag in zip(current["tokens"], current["tags"])
    )


def tag_card(space: dict, current: dict, tags: list[str]) -> None:
    saved = space["labels"].get(space["position"])
    changed = saved is not None and (saved["tokens"], saved["tags"]) != (current["tokens"], current["tags"])
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Tags")
            saved_badge(saved is not None and not changed)
        st.caption("Select tokens on the left, then pick a tag.")
        selection = space["view"].get("selection", [])
        with st.container(horizontal=True, gap="small"):
            for index, tag in enumerate(tags):
                st.button(
                    f":{tag_color(tags, tag)}[{plain_text(tag)}]", key=f"token_tag_{index}",
                    on_click=transact, args=(TASK, apply_tag, tag), disabled=not selection,
                )
        with st.container(horizontal=True):
            st.button("Select untagged", icon=":material/select_all:", key="token_select_default", on_click=transact, args=(TASK, select_default))
            st.button("Save and next", icon=":material/check:", key="token_save", type="primary", on_click=commit, args=(TASK, save), disabled=not current["tokens"])
        with st.container(horizontal=True):
            st.button("Discard edits", icon=":material/restart_alt:", key="token_discard", on_click=transact, args=(TASK, discard))
            st.button("Remove label", icon=":material/delete:", key="token_unlabel", on_click=commit, args=(TASK, unlabel), disabled=saved is None)
        navigation(TASK)


def body(space: dict) -> None:
    if not space["config"]["field"]:
        st.caption("Choose the text field in the sidebar.")
        return
    tags = tag_set(space["config"])
    current = draft(space)
    position, size = space["position"], len(space["records"])
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader(f"Tokens, row {position + 1} of {size}")
            with st.container(height=CONTEXT_HEIGHT):
                if current["tokens"]:
                    key = widget_key(space, NAME, "selection")
                    st.pills(
                        "Tokens", list(range(len(current["tokens"]))), selection_mode="multi", key=key,
                        default=space["view"].get("selection", []),
                        format_func=lambda i: token_label(current["tokens"][i], current["tags"][i], tags),
                        on_change=transact, args=(TASK, set_selection, key), label_visibility="collapsed",
                    )
                else:
                    st.caption("This row has no text.")
        with st.container(border=True):
            st.subheader("Tagged text")
            with st.container(height=PREVIEW_HEIGHT):
                st.markdown(tagged_text(current, tags) or "No tokens.")
    with right:
        tag_card(space, current, tags)
        counts = Counter(tag for label in space["labels"].values() for tag in label["tags"] if tag != tags[0])
        names = list(dict.fromkeys([*tags[1:], *counts]))
        distribution_card(pd.DataFrame({"name": names, "count": [counts[name] for name in names]}))
    records_table(space, lambda label: " ".join(f"{w}/{t}" for w, t in zip(label["tokens"], label["tags"]) if t != tags[0]))


def render() -> None:
    page(TASK)
