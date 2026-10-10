"""Small shared helpers (timestamps, slugs, confidence checks)."""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any


SCHEMA_VERSION = 1


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def slugify(name: str, default: str = "item") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s or default


def is_confidence(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and not math.isnan(float(x))
        and 0.0 <= float(x) <= 1.0
    )
