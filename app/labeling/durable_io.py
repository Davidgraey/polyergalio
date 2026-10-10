"""Durable file I/O: canonical JSON, CRC32, atomic write-fsync-rename, tolerant JSON reads."""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def crc32(obj: Any) -> int:
    return zlib.crc32(canonical(obj).encode("utf-8")) & 0xFFFFFFFF


def fsync_dir(path: str) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(d)


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def read_json(path: str, default: Any) -> Any:
    """Read JSON; a corrupt file is moved aside (never overwritten) and default returned."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError):
        os.replace(path, f"{path}.corrupt-{int(time.time())}")
        return default
