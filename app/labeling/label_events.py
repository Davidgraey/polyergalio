"""Core data types: task definitions, label events, tokenization, validation.

Every label carries a ``value`` and a ``confidence`` (annotator confidence in
[0, 1]). Item-level labels (tokens, spans, contrastive pairs) each carry their
own confidence as well; the event-level confidence is their mean.

Value shapes by task type
-------------------------
classification (single) : {"label": str}
classification (multi)  : {"labels": [str, ...]}
token                   : {"tokenizer": str,
                           "tokens": [{"i", "text", "start", "end", "tag", "confidence"}]}
span                    : {"tokenizer": str,
                           "spans": [{"start", "end", "text", "label", "confidence",
                                      "token_start", "token_end"}]}
contrastive             : {"anchor_id": str,
                           "pairs": [{"sample_id", "relation", "confidence"}]}

Character offsets are stored alongside the covered text so every label can be
re-verified against the dataset (see ``validate_label``).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from task_defs import OUTSIDE_TAG, TaskDef, TaskType
from tokenizing import tokenize
from util import SCHEMA_VERSION, is_confidence, now_iso


@dataclass
class LabelEvent:
    """One record in the append-only label log.

    Attributes
    ----------
    op : str
        ``"label"`` or ``"retract"``.
    model_score : float, optional
        Model uncertainty at labeling time.
    supersedes : str, optional
        ``event_id`` of the record this one replaces.
    seq : int
        Assigned by the store.
    """

    sample_id: str
    task_id: str
    task_type: str
    value: Any
    confidence: float
    annotator: str
    mode: str
    op: str = "label"
    model_key: Optional[str] = None
    model_version: Optional[int] = None
    model_score: Optional[float] = None
    supersedes: Optional[str] = None
    note: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=now_iso)
    seq: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_label(task: TaskDef, text: str, value: Any, confidence: Any) -> List[str]:
    """Return the problems with a label value; empty when it is valid.

    Overlapping and nested spans are allowed; only exact repeats (same offsets and label) are not.
    """
    errs: List[str] = []
    if not is_confidence(confidence):
        errs.append("Confidence must be a number in [0, 1].")
    if not isinstance(value, dict):
        return errs + ["Value must be an object."]

    t = task.type
    if t is TaskType.CLASSIFICATION:
        if task.multi_label:
            labs = value.get("labels")
            if not isinstance(labs, list):
                errs.append("Multi-label value needs a 'labels' list.")
            else:
                bad = [x for x in labs if x not in task.labels]
                if bad:
                    errs.append(f"Unknown labels: {bad}")
                if len(set(labs)) != len(labs):
                    errs.append("Duplicate labels.")
        elif value.get("label") not in task.labels:
            errs.append("Pick a label.")

    elif t is TaskType.TOKEN:
        toks = value.get("tokens")
        expected = tokenize(text)
        if not isinstance(toks, list) or not toks:
            errs.append("Token value needs a non-empty 'tokens' list.")
        else:
            if len(toks) != len(expected):
                errs.append(f"Expected {len(expected)} tokens, got {len(toks)}.")
            for j, tk in enumerate(toks):
                try:
                    s, e = int(tk["start"]), int(tk["end"])
                except (KeyError, TypeError, ValueError):
                    errs.append(f"Token {j}: missing offsets.")
                    continue
                if text[s:e] != tk.get("text"):
                    errs.append(f"Token {j}: offsets do not match text.")
                if tk.get("tag") not in task.labels:
                    errs.append(f"Token {j}: unknown tag {tk.get('tag')!r}.")
                if not is_confidence(tk.get("confidence")):
                    errs.append(f"Token {j}: confidence must be in [0, 1].")

    elif t is TaskType.SPAN:
        spans = value.get("spans")
        if not isinstance(spans, list):
            errs.append("Span value needs a 'spans' list.")
        else:
            seen = set()
            for j, sp in enumerate(spans):
                try:
                    s, e = int(sp["start"]), int(sp["end"])
                except (KeyError, TypeError, ValueError):
                    errs.append(f"Span {j}: missing offsets.")
                    continue
                if not (0 <= s < e <= len(text)):
                    errs.append(f"Span {j}: offsets out of range.")
                elif text[s:e] != sp.get("text"):
                    errs.append(f"Span {j}: offsets do not match text.")
                if sp.get("label") not in task.labels:
                    errs.append(f"Span {j}: unknown label {sp.get('label')!r}.")
                if not is_confidence(sp.get("confidence")):
                    errs.append(f"Span {j}: confidence must be in [0, 1].")
                k = (s, e, sp.get("label"))
                if k in seen:
                    errs.append(f"Span {j}: duplicate.")
                seen.add(k)

    elif t is TaskType.CONTRASTIVE:
        anchor = value.get("anchor_id")
        pairs = value.get("pairs")
        if not anchor:
            errs.append("Missing anchor_id.")
        if not isinstance(pairs, list) or not pairs:
            errs.append("Contrastive value needs a non-empty 'pairs' list.")
        else:
            ids = [p.get("sample_id") for p in pairs]
            if len(set(ids)) != len(ids):
                errs.append("Duplicate candidates.")
            for j, p in enumerate(pairs):
                if p.get("sample_id") == anchor:
                    errs.append(f"Pair {j}: candidate equals anchor.")
                if p.get("relation") not in task.labels:
                    errs.append(f"Pair {j}: choose a relation.")
                if not is_confidence(p.get("confidence")):
                    errs.append(f"Pair {j}: confidence must be in [0, 1].")
    return errs


def summarize_value(task_type: str, value: Any) -> str:
    """Short human-readable summary for tables."""
    if value is None:
        return "—"
    try:
        if task_type == TaskType.CLASSIFICATION.value:
            if "labels" in value:
                return ", ".join(value["labels"]) or "(none)"
            return str(value.get("label"))
        if task_type == TaskType.TOKEN.value:
            toks = value.get("tokens", [])
            n = sum(1 for t in toks if t.get("tag") != OUTSIDE_TAG)
            return f"{n} tagged / {len(toks)} tokens"
        if task_type == TaskType.SPAN.value:
            spans = value.get("spans", [])
            if not spans:
                return "(no spans)"
            s = "; ".join(f"{sp['label']}: {sp['text']}" for sp in spans)
            return s if len(s) <= 80 else s[:77] + "…"
        if task_type == TaskType.CONTRASTIVE.value:
            counts: Dict[str, int] = {}
            for p in value.get("pairs", []):
                counts[p.get("relation")] = counts.get(p.get("relation"), 0) + 1
            return ", ".join(f"{k}×{v}" for k, v in counts.items())
    except Exception:
        pass
    return str(value)[:80]
