"""Task definitions: the base ``TaskDef`` (type, labels, label definitions, rubric,
column sources) and the built-in default tasks.

Column convention (``{task}`` is the task id, a stable slug of the name):
``{task}_labels`` holds the label as {"value", "confidence"}; ``{task}_pred``
holds the model's raw output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


OUTSIDE_TAG = "O"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    TOKEN = "token"
    SPAN = "span"
    CONTRASTIVE = "contrastive"

    @property
    def display(self) -> str:
        return {
            "classification": "Full-sample classification",
            "token": "Token labeling",
            "span": "Span labeling (start / end + label)",
            "contrastive": "Contrastive (vs. anchor)",
        }[self.value]


class Mode(str, Enum):
    INITIAL = "initial"
    RELABEL = "relabel"


@dataclass
class TaskDef:
    """What is labeled, how, and where it lives.

    Attributes
    ----------
    multi_label : bool
        Classification only.
    n_candidates : int
        Contrastive only.
    archived : bool
        Hidden from the UI; labels are never deleted.
    rubric : str
        Labeling instructions shown beside the labeling controls.
    label_definitions : dict
        Label to its meaning, shown with the rubric.
    label_col, pred_col : str
        Optional overrides of the label and prediction column names.
    """

    id: str
    name: str
    type: TaskType
    labels: List[str]
    multi_label: bool = False
    n_candidates: int = 4
    archived: bool = False
    rubric: str = ""
    label_definitions: Dict[str, str] = field(default_factory=dict)
    label_col: str = ""
    pred_col: str = ""

    def definition_of(self, label: str) -> str:
        return (self.label_definitions or {}).get(label, "")

    def missing_definitions(self) -> List[str]:
        return [x for x in self.labels if not self.definition_of(x).strip()]

    @property
    def labels_column(self) -> str:
        return self.label_col or f"{self.id}_labels"

    @property
    def pred_column(self) -> str:
        return self.pred_col or f"{self.id}_pred"

    def definition(self) -> Dict[str, Any]:
        """The task contract: what is labeled, with which labels, in which columns."""
        return {"task": self.id, "name": self.name, "type": self.type.value, "labels": list(self.labels),
                "multi_label": self.multi_label, "labels_column": self.labels_column,
                "pred_column": self.pred_column, "rubric": self.rubric,
                "label_definitions": dict(self.label_definitions)}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskDef":
        d = dict(d)
        d["type"] = TaskType(d["type"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


DEFAULT_TASKS: List[TaskDef] = [
    TaskDef("sentiment", "Sentiment", TaskType.CLASSIFICATION,
            ["negative", "neutral", "positive"],
            rubric="Judge the customer's overall feeling about the whole review, not any single sentence. "
                   "If feelings are mixed, choose the one that dominates. A polite closing ('thanks') does "
                   "not make an otherwise negative review positive.",
            label_definitions={
                "negative": "The customer is dissatisfied, complaining, or describing a failure.",
                "neutral": "Factual or balanced, with no clear satisfaction or dissatisfaction.",
                "positive": "The customer is satisfied, pleased, or recommending.",
            }),
    TaskDef("ner", "Named entities", TaskType.TOKEN,
            [OUTSIDE_TAG, "PER", "ORG", "LOC", "PRODUCT"],
            rubric="Tag every token. Use 'O' for anything that is not a named entity. Tag each token of a "
                   "multi-word name (e.g. both words of a company name). Do not tag generic nouns.",
            label_definitions={
                OUTSIDE_TAG: "Not part of any named entity.",
                "PER": "A named person (first name, surname, or full name).",
                "ORG": "A named company, team, or institution.",
                "LOC": "A named place: city, country, address, or region.",
                "PRODUCT": "A named product, model, or service.",
            }),
    TaskDef("aspects", "Aspect spans", TaskType.SPAN,
            ["DELIVERY", "PRICE", "QUALITY", "SUPPORT"],
            rubric="Select the shortest phrase that expresses an opinion or fact about an aspect. Spans may "
                   "overlap when one phrase covers two aspects. If the text mentions none, save with no spans.",
            label_definitions={
                "DELIVERY": "Shipping speed, packaging condition, courier, or arrival problems.",
                "PRICE": "Cost, value for money, discounts, fees, or refunds.",
                "QUALITY": "How well the product is made or works.",
                "SUPPORT": "Interaction with customer service or help channels.",
            }),
    TaskDef("similarity", "Similar to anchor", TaskType.CONTRASTIVE,
            ["similar", "dissimilar"], n_candidates=4,
            rubric="Compare each candidate with the anchor sample above. Decide on the underlying issue, not "
                   "on shared words or tone.",
            label_definitions={
                "similar": "Describes the same underlying issue or topic as the anchor.",
                "dissimilar": "Describes a different issue or topic from the anchor.",
            }),
]
