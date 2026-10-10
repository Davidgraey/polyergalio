"""
------------------------ Constants & lookups for Pipeline and processor / encoders ------------------------------
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict


# -------------------------Time -> ratios -------------------------
class Period(Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


DELTA_LOOKUP = {
    "minute": 60,
    "hour": 3600,
    "day": 3600 * 24,
    "week": 3600 * 24 * 7,
    "month": 3600 * 24 * 30.437,  # avg days per month
    "year": 3600 * 24 * 365.25,  # avg days per year
}


SPECIAL_TOKEN_PIECES: Dict[str, str] = {
    "PAD": "<pad>",
    "BOS": "<s>",
    "EOS": "</s>",
    "UNK": "<unk>",
    "CLS": "<cls>",
    "SEP": "<sep>",
    "MARK": "<mark>",
    "MASK": "<mask>",
    "INSTRUCTION": "<instruction>",
    "END_INSTRUCTION": "</instruction>",
    "SYSTEM": "<system>",
    "END_SYSTEM": "</system>",
    "USER": "<user>",
    "END_USER": "</user>",
    "AGENT": "<agent>",
    "END_AGENT": "</agent>",
    "QUESTION": "<question>",
    "END_QUESTION": "</question>",
    "ANSWER": "<answer>",
    "END_ANSWER": "</answer>",
}

CORE_FIELDS = ("PAD", "BOS", "EOS", "UNK")

@dataclass
class SpecialTokens:
    """Special token identifiers and their values."""

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    CLS: int = 3
    SEP: int = 4
    MARK: int = 5
    UNK: int = 6
    MASK: int = 7
    INSTRUCTION: int = 8
    END_INSTRUCTION: int = 9
    SYSTEM: int = 10
    END_SYSTEM: int = 11
    USER: int = 12
    END_USER: int = 13
    AGENT: int = 14
    END_AGENT: int = 15
    QUESTION: int = 16
    END_QUESTION: int = 17
    ANSWER: int = 18
    END_ANSWER: int = 19
    TOKEN_OFFSET: int = 20

    @property
    def ids(self) -> frozenset:
        """Every special token id, TOKEN_OFFSET excluded."""
        return frozenset(value for name, value in asdict(self).items() if name != "TOKEN_OFFSET")
