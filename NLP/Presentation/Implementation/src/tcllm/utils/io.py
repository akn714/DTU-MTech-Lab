"""Filesystem helpers: atomic writes, JSONL round-tripping, content fingerprints.

Everything that writes under ``artifacts/`` goes through :func:`atomic_write_text`.
An interrupted run must never leave a half-written file behind: a truncated
predictions file that still parses as JSON is worse than no file at all, because
the resume logic would trust it and skip the work.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "atomic_write_text",
    "read_json",
    "write_json",
    "read_jsonl",
    "iter_jsonl",
    "write_jsonl",
    "sha256_text",
    "fingerprint_rows",
]


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically (temp file in the same directory + rename).

    The temp file is created in the destination directory so that ``os.replace``
    stays on one filesystem and is therefore genuinely atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Write ``obj`` as UTF-8 JSON. ``ensure_ascii=False`` keeps Arabic and Hindi readable."""
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    """Write one JSON object per line, UTF-8, ``\\n`` terminated."""
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return atomic_write_text(path, body)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield rows lazily. Blank lines are skipped; malformed lines raise with the line number."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_rows(rows: Iterable[dict], keys: tuple[str, ...] = ("text", "label")) -> str:
    """Order-independent fingerprint of a row collection.

    Used to record *which* data produced a result, and to detect a test set that
    changed underneath a cached prediction. Sorting the per-row hashes means a
    reshuffle does not change the fingerprint — only a change in content does.
    """
    per_row = sorted(sha256_text("\x1f".join(str(row.get(k, "")) for k in keys)) for row in rows)
    digest = hashlib.sha256()
    for h in per_row:
        digest.update(h.encode("ascii"))
    return digest.hexdigest()
