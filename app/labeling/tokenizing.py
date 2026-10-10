"""Word/punctuation tokenizer. Label offsets depend on it, so changing it invalidates saved token and span labels."""

from __future__ import annotations

import re
from typing import List, Tuple


TOKENIZER_ID = "regex-word-punct-v1"


_TOKEN_RE = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Word/punctuation tokens as (text, start_char, end_char)."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]
