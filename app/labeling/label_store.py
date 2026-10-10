"""Crash-safe label persistence.

Design
------
* **Append-only write-ahead log** (``labels.wal.jsonl``) is the source of
  truth. Every save is one JSON line, written with ``flush`` + ``fsync`` before
  the UI confirms it, so a crash loses at most the label being typed.
* Each line carries a **CRC32** of its canonical JSON. On load, unparseable or
  checksum-failing lines (e.g. a half-written tail after a power cut) are
  **quarantined** to ``quarantine.jsonl`` for manual review — the WAL itself is
  never rewritten or truncated.
* **Nothing is ever deleted.** Relabels append a new event that ``supersedes``
  the old one; removals append a ``retract`` event. Full history is auditable.
* **Snapshots** of the full event history are written atomically every N
  events (and on demand) into ``snapshots/``, rotated. If the WAL is lost the
  store recovers from the newest valid snapshot and rebuilds the WAL.
* **Drafts** (multi-step work such as spans being added) are persisted
  atomically on every change, so unfinished work survives a restart.
* Config files (tasks, manifest) use atomic write-temp-fsync-rename.

Single-process use is assumed; an in-process lock plus ``flock`` on the WAL
guards concurrent sessions of the same Streamlit server.
"""

from __future__ import annotations

import glob
import json
import os
import threading
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from durable_io import atomic_write_bytes, atomic_write_json, canonical, crc32, fsync_dir, read_json
from label_events import LabelEvent
from util import SCHEMA_VERSION, now_iso

try:
    import fcntl
except ImportError:
    fcntl = None


Key = Tuple[str, str]


class LabelStore:
    WAL = "labels.wal.jsonl"
    QUARANTINE = "quarantine.jsonl"
    DRAFTS = "drafts.json"
    SNAP_DIR = "snapshots"

    def __init__(self, root: str, snapshot_every: int = 25, keep_snapshots: int = 5):
        self.root = root
        self.snapshot_every = snapshot_every
        self.keep_snapshots = keep_snapshots
        os.makedirs(self.snap_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._events: List[dict] = []
        self._current: Dict[Key, dict] = {}
        self._history: Dict[Key, List[dict]] = defaultdict(list)
        self.seq = 0
        self._since_snapshot = 0
        self._needs_newline = False
        self.last_write_ts: Optional[str] = None
        self.last_snapshot_seq: Optional[int] = None
        self.report: Dict[str, Any] = {
            "events_loaded": 0, "quarantined": 0,
            "truncated_tail": False, "recovered_from_snapshot": None,
        }
        self._drafts: Dict[str, Any] = read_json(self.drafts_path, {})
        self._load()

    @property
    def wal_path(self) -> str:
        return os.path.join(self.root, self.WAL)

    @property
    def quarantine_path(self) -> str:
        return os.path.join(self.root, self.QUARANTINE)

    @property
    def drafts_path(self) -> str:
        return os.path.join(self.root, self.DRAFTS)

    @property
    def snap_dir(self) -> str:
        return os.path.join(self.root, self.SNAP_DIR)

    def _load(self) -> None:
        if os.path.exists(self.wal_path):
            with open(self.wal_path, "rb") as fh:
                data = fh.read()
            if data and not data.endswith(b"\n"):
                self._needs_newline = True
                self.report["truncated_tail"] = True
            bad = []
            for lineno, raw in enumerate(data.split(b"\n"), 1):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8"))
                    if not isinstance(rec, dict):
                        raise ValueError("not a JSON object")
                    crc = rec.pop("crc", None)
                    if crc != crc32(rec):
                        raise ValueError("checksum mismatch")
                    for k in ("seq", "sample_id", "task_id", "op", "event_id"):
                        if k not in rec:
                            raise ValueError(f"missing field {k!r}")
                except Exception as e:
                    bad.append({"line": lineno, "error": str(e),
                                "raw": raw.decode("utf-8", "replace")})
                    continue
                self._apply(rec)
            self.report["events_loaded"] = len(self._events)
            self.report["quarantined"] = len(bad)
            if bad:
                atomic_write_bytes(
                    self.quarantine_path,
                    "".join(json.dumps(b, ensure_ascii=False) + "\n" for b in bad).encode("utf-8"),
                )
        else:
            snap = self._latest_valid_snapshot()
            if snap is not None:
                path, payload = snap
                for rec in payload["events"]:
                    self._apply(rec)
                atomic_write_bytes(self.wal_path, "".join(
                    canonical({**r, "crc": crc32(r)}) + "\n" for r in self._events
                ).encode("utf-8"))
                self.report["recovered_from_snapshot"] = os.path.basename(path)
                self.report["events_loaded"] = len(self._events)
        snaps = self._snapshot_files()
        if snaps:
            self.last_snapshot_seq = int(os.path.basename(snaps[-1])[9:19])

    def _apply(self, rec: dict) -> None:
        key = (rec["sample_id"], rec["task_id"])
        self._events.append(rec)
        self._history[key].append(rec)
        if rec["op"] == "retract":
            self._current.pop(key, None)
        else:
            self._current[key] = rec
        self.seq = max(self.seq, int(rec["seq"]))
        self.last_write_ts = rec.get("ts", self.last_write_ts)

    def append(self, event: LabelEvent) -> dict:
        """Durably append one event.

        Returns
        -------
        dict
            The stored record.

        Raises
        ------
        OSError
            If the write fails. In-memory state is only updated after the fsync succeeds.
        """
        with self._lock:
            rec = event.to_dict()
            rec["seq"] = self.seq + 1
            line = canonical({**rec, "crc": crc32(rec)}) + "\n"
            created = not os.path.exists(self.wal_path)
            with open(self.wal_path, "ab") as fh:
                if fcntl:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    if self._needs_newline:
                        fh.write(b"\n")
                    fh.write(line.encode("utf-8"))
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    if fcntl:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            self._needs_newline = False
            if created:
                fsync_dir(self.root)
            self._apply(rec)
            self._since_snapshot += 1
            if self._since_snapshot >= self.snapshot_every:
                self.snapshot()
            return rec

    def retract(self, sample_id: str, task_id: str, task_type: str,
                annotator: str, mode: str, note: str = "") -> Optional[dict]:
        cur = self.current(sample_id, task_id)
        if cur is None:
            return None
        return self.append(LabelEvent(
            sample_id=sample_id, task_id=task_id, task_type=task_type,
            value=None, confidence=0.0, annotator=annotator, mode=mode,
            op="retract", supersedes=cur["event_id"], note=note,
        ))

    def _snapshot_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.snap_dir, "snapshot-*.json")))

    def snapshot(self, force: bool = False) -> Optional[str]:
        with self._lock:
            if not force and self.last_snapshot_seq == self.seq:
                return None
            payload = {
                "schema_version": SCHEMA_VERSION, "seq": self.seq, "ts": now_iso(),
                "n_events": len(self._events), "n_current": len(self._current),
                "events": self._events,
            }
            payload["crc"] = crc32({k: v for k, v in payload.items() if k != "crc"})
            path = os.path.join(self.snap_dir, f"snapshot-{self.seq:010d}.json")
            atomic_write_json(path, payload)
            for old in self._snapshot_files()[:-self.keep_snapshots]:
                os.remove(old)
            self.last_snapshot_seq = self.seq
            self._since_snapshot = 0
            return path

    def _latest_valid_snapshot(self):
        for path in reversed(self._snapshot_files()):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                crc = payload.pop("crc", None)
                if crc == crc32(payload):
                    return path, payload
            except Exception:
                continue
        return None

    def get_draft(self, key: str) -> Any:
        d = self._drafts.get(key)
        return None if d is None else d.get("value")

    def save_draft(self, key: str, value: Any) -> None:
        with self._lock:
            self._drafts[key] = {"value": value, "ts": now_iso()}
            atomic_write_json(self.drafts_path, self._drafts)

    def clear_draft(self, key: str) -> None:
        with self._lock:
            if self._drafts.pop(key, None) is not None:
                atomic_write_json(self.drafts_path, self._drafts)

    def current(self, sample_id: str, task_id: str) -> Optional[dict]:
        return self._current.get((sample_id, task_id))

    def history(self, sample_id: str, task_id: str) -> List[dict]:
        return list(self._history.get((sample_id, task_id), []))

    def current_for_task(self, task_id: str) -> Dict[str, dict]:
        return {sid: rec for (sid, tid), rec in self._current.items() if tid == task_id}

    def events(self, limit: Optional[int] = None) -> List[dict]:
        ev = self._events[::-1]
        return ev if limit is None else ev[:limit]

    @property
    def n_events(self) -> int:
        return len(self._events)

    @property
    def n_current(self) -> int:
        return len(self._current)

    def all_current(self) -> Iterable[dict]:
        return list(self._current.values())
