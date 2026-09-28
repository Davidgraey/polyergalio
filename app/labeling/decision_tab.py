"""Decision labeling: one BINARY, CHOICE or SCORE question over each structured record."""

import json
import sys

import pandas as pd
import streamlit as st
from label_common import (
    CONTEXT_HEIGHT,
    MAX_SEGMENTED,
    PREVIEW_HEIGHT,
    IntegrityError,
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
    step,
    widget_key,
    workspace,
)
from label_tasks import DECISION_TASKS
from polyergalio.models.constants import DECISION_TYPES

NAME = "decision"
TASK = sys.modules[__name__]
EXAMPLES = DECISION_TASKS
BINARY_OPTIONS = ["false", "true"]
KINDS = [member.name for member in DECISION_TYPES]
DEFAULT_CONFIG = dict(kind="CHOICE", instructions="", options_text="", levels=4, columns=[], autosave="")


def question_options(kind: str, options_text: str, levels: int) -> list[str]:
    """Option texts in answer-index order, fixed for BINARY and SCORE."""
    if kind == "BINARY":
        return list(BINARY_OPTIONS)
    if kind == "SCORE":
        return [str(level) for level in range(int(levels))]
    return lines(options_text)


def question(config: dict) -> dict:
    return dict(
        kind=config["kind"].lower(),
        instructions=config["instructions"],
        options=question_options(config["kind"], config["options_text"], config["levels"]),
        columns=list(config["columns"]),
    )


def record_state(record: dict, columns: list[str]) -> dict:
    return {column: record.get(column) for column in columns}


def sequence_preview(q: dict, state: dict) -> str:
    """Model input as lines: type and instructions, one line per option, then the state."""
    header = f"{q['kind']} {q['instructions']}".strip()
    options = "\n".join(f"{index}: {option}" for index, option in enumerate(q["options"]))
    return f"{header}\n\n{options}\n\n{json.dumps(state, ensure_ascii=False)}"


# -------------    label contract    -----------------------------
def validate_label(label) -> list[str]:
    if not isinstance(label, dict):
        return ["label is not an object"]
    problems = []
    kinds = [kind.lower() for kind in KINDS]
    if label.get("kind") not in kinds:
        problems.append(f"decision type {label.get('kind')!r} is not one of {kinds}")
    if not isinstance(label.get("instructions"), str):
        problems.append("instructions are not text")
    options = label.get("options")
    if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, str) and o for o in options):
        problems.append("options must be at least two non-empty strings")
        options = []
    answer = label.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < max(len(options), 1):
        problems.append(f"answer {answer!r} is not an option index")
    elif options and label.get("option") != options[answer]:
        problems.append("option text does not match the answer index")
    if label.get("kind") == "binary" and options and options != BINARY_OPTIONS:
        problems.append("binary options must be [false, true]")
    if label.get("kind") == "score" and options and options != [str(i) for i in range(len(options))]:
        problems.append("score options must be the levels 0..n-1")
    if not isinstance(label.get("record"), dict):
        problems.append("the sample record is missing")
    if not isinstance(label.get("columns"), list) or not isinstance(label.get("state"), dict):
        problems.append("state or columns are missing")
    elif isinstance(label.get("record"), dict) and label["state"] != record_state(label["record"], label["columns"]):
        problems.append("state does not match the record")
    return problems


def to_row(label: dict) -> dict:
    return {
        "decisiontype": label["kind"],
        "instructions": label["instructions"],
        "options": label["options"],
        "state": label["state"],
        "answer": label["answer"],
        "option": label["option"],
        "record": label["record"],
    }


def from_row(row: dict) -> dict:
    if not isinstance(row.get("state"), dict):
        raise IntegrityError("state is not an object")
    record = row.get("record", row["state"])
    return {
        "record": record,
        "kind": row["decisiontype"],
        "instructions": row["instructions"],
        "options": row["options"],
        "columns": list(row["state"]),
        "state": row["state"],
        "answer": row["answer"],
        "option": row["options"][row["answer"]] if isinstance(row["answer"], int) and 0 <= row["answer"] < len(row["options"]) else None,
    }


def is_labeled_row(row: dict) -> bool:
    return {"decisiontype", "options", "state", "answer"} <= row.keys()


def config_for_example(example: dict, records: list[dict], current: dict) -> dict:
    return {
        **current, "kind": example["kind"], "instructions": example["instructions"],
        "options_text": "\n".join(example["options"]), "levels": max(2, len(example["options"])),
        "columns": record_columns(records),
    }


def config_for_records(records: list[dict], current: dict) -> dict:
    return {**current, "columns": record_columns(records)}


def config_from_labels(labels: dict, current: dict) -> dict:
    first = labels[min(labels)]
    return {
        **current, "kind": first["kind"].upper(), "instructions": first["instructions"],
        "options_text": "\n".join(first["options"]), "levels": max(2, len(first["options"])),
        "columns": first["columns"],
    }


# -------------    actions    ------------------------------------
def unlabel(space: dict) -> None:
    space["labels"].pop(space["position"], None)
    bump(space)


def choose(space: dict, key: str) -> None:
    answer = st.session_state[key]
    position = space["position"]
    if answer is None:
        space["labels"].pop(position, None)
        return
    q = question(space["config"])
    record = space["records"][position]
    space["labels"][position] = {
        "record": record, **q, "state": record_state(record, q["columns"]),
        "answer": int(answer), "option": q["options"][int(answer)],
    }
    step(space, 1)


# -------------    layout    -------------------------------------
def sidebar() -> None:
    space = workspace(TASK)
    data_section(TASK)
    with st.expander("Task", expanded=True, icon=":material/tune:"):
        config_widget(TASK, st.selectbox, "Type", "kind", options=KINDS, format_func=str.capitalize)
        config_widget(TASK, st.text_area, "Instructions", "instructions", height=68)
        if space["config"]["kind"] == "CHOICE":
            config_widget(TASK, st.text_area, "Options, one per line", "options_text", height=68)
        elif space["config"]["kind"] == "SCORE":
            config_widget(TASK, st.number_input, "Levels", "levels", clean=int, min_value=2, max_value=100)
        config_widget(TASK, st.multiselect, "State fields", "columns", clean=list, options=record_columns(space["records"]))


def answer_card(space: dict, q: dict) -> None:
    saved = space["labels"].get(space["position"])
    matches = saved is not None and all(saved[k] == q[k] for k in ("kind", "instructions", "options", "columns"))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Answer")
            st.badge(q["kind"], color="blue")
            saved_badge(saved is not None)
        st.write(q["instructions"] or "No instructions.")
        if saved is not None and not matches:
            st.caption(f"Saved under a different question as {plain_text(saved['option'])}. Answering replaces it.")
        if len(q["options"]) < 2:
            st.warning("Enter at least two options.", icon=":material/warning:")
        else:
            key = widget_key(space, NAME, "answer")
            settings = dict(
                options=list(range(len(q["options"]))), format_func=q["options"].__getitem__,
                key=key, on_change=commit, args=(TASK, choose, key), label_visibility="collapsed",
            )
            answered = saved["answer"] if matches else None
            if q["kind"] == "choice" or len(q["options"]) > MAX_SEGMENTED:
                st.radio("Answer", index=answered, **settings)
            else:
                st.segmented_control("Answer", default=answered, width="stretch", **settings)
            st.caption("Picking an option saves the label and moves to the next row.")
        st.button(
            "Remove label", icon=":material/delete:", key="decision_unlabel",
            on_click=commit, args=(TASK, unlabel), disabled=saved is None,
        )
        navigation(TASK)


def body(space: dict) -> None:
    q = question(space["config"])
    position, size = space["position"], len(space["records"])
    state = record_state(space["records"][position], q["columns"])
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader(f"Context, row {position + 1} of {size}")
            with st.container(height=CONTEXT_HEIGHT):
                if state:
                    st.table({plain_text(k): plain_text(v) for k, v in state.items()}, border="horizontal")
                else:
                    st.caption("No state fields selected.")
        with st.container(border=True):
            st.subheader("Model input")
            with st.container(height=PREVIEW_HEIGHT):
                st.code(sequence_preview(q, state), language="text", wrap_lines=True)
    with right:
        answer_card(space, q)
        chosen = [label["option"] for label in space["labels"].values()]
        names = list(dict.fromkeys([*q["options"], *chosen]))
        distribution_card(pd.DataFrame({"name": names, "count": [chosen.count(n) for n in names]}))
    records_table(space, lambda label: label["option"])


def render() -> None:
    page(TASK)
