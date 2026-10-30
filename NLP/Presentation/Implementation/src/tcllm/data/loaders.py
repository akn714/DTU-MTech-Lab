"""Dataset loading.

Phase 1 implements SENTIMENT (project rule 5). Nothing here is hard-coded: the
HuggingFace id, the pinned revision, the file layout, the label names and the
expected split sizes all come from ``configs/datasets.yaml``. Adding TAXI1500,
INTENT or SCENARIO in Phase 7 is therefore a config entry plus a parse function,
not a rewrite.

Loading is offline after the first run: files are cached under
``artifacts/data/raw/hf_cache`` at a pinned revision, so a re-run weeks later
reads byte-identical data even if upstream moves.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ..config import get_dataset_spec, processed_dir, raw_cache_dir, resolve_language
from ..utils.io import read_json, read_jsonl, write_json, write_jsonl
from .splits import SPLIT_NAMES, Example, SplitBundle, assert_official_sizes

# A literal backslash, as it appears in the raw source data.
BACKSLASH = chr(92)

# The upstream JSONL is double-escaped JSON: the file contains \\u2019, so a
# single json.loads yields the six characters ’ rather than the character ’.
# This pattern matches the surviving sequence, and is also what the audit counts.
_UNICODE_ESCAPE = re.compile(re.escape(BACKSLASH) + r"u([0-9a-fA-F]{4})")

# Any remaining backslash followed by a character. Used only to *report* what was
# preserved, never to rewrite it: this dataset contains genuine kaomoji such as
# \(^-^ )/ and \m/ which would be corrupted by blanket unescaping.
_ANY_BACKSLASH = re.compile(re.escape(BACKSLASH) + r"(.)", re.DOTALL)

# Tasks with a loader. The rest are declared in configs/datasets.yaml but commented out.
_IMPLEMENTED_TASKS = ("sentiment",)


# --------------------------------------------------------------------------- #
# text cleaning
# --------------------------------------------------------------------------- #
def _recombine_surrogates(text: str) -> str:
    """Join surrogate pairs that decoding \\uD83D and \\uDE00 separately would leave split.

    Tweets contain emoji, which are surrogate pairs in UTF-16. Decoding each half
    independently produces two lone surrogates, which are not valid text and crash
    JSON encoding downstream.
    """
    if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return text
    try:
        return text.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def decode_escapes(
    text: str,
    *,
    unescape_unicode: bool = True,
    unescape_quotes: bool = True,
) -> tuple[str, dict[str, int]]:
    """Undo the upstream double-escaping. Returns ``(cleaned_text, counts)``.

    Only two transformations are applied, both counted for the audit:

    * ``\\uXXXX`` -> the character it denotes (``\\u2019`` -> ``’``, ``\\u002c`` -> ``,``)
    * ``\\"`` -> ``"``

    Every other backslash is preserved on purpose. ``\\(^-^ )/`` and ``\\m/`` are
    real content, not escaping artefacts, and a blanket unescape would destroy them.
    """
    counts = {"unicode_escapes": 0, "escaped_quotes": 0, "backslashes_preserved": 0}
    if not text:
        return text, counts

    if unescape_unicode and BACKSLASH + "u" in text:
        counts["unicode_escapes"] = len(_UNICODE_ESCAPE.findall(text))
        text = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
        text = _recombine_surrogates(text)

    if unescape_quotes:
        target = BACKSLASH + '"'
        found = text.count(target)
        if found:
            counts["escaped_quotes"] = found
            text = text.replace(target, '"')

    counts["backslashes_preserved"] = len(_ANY_BACKSLASH.findall(text))
    return text, counts


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def fetch_split_files(
    spec: dict[str, Any],
    language_key: str,
    *,
    artifacts: str | Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download (or reuse cached) upstream files for all three splits."""
    template = spec["hf_file_template"]
    cache = raw_cache_dir(artifacts)
    files: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        path = hf_hub_download(
            repo_id=spec["hf_id"],
            filename=template.format(language=language_key, split=split),
            repo_type="dataset",
            revision=spec.get("hf_revision"),
            cache_dir=str(cache),
            force_download=force,
        )
        files[split] = Path(path)
    return files


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_split(
    path: Path,
    spec: dict[str, Any],
) -> tuple[list[Example], dict[str, int]]:
    """Parse one upstream JSONL file into Examples, cleaning text as configured."""
    labels: list[str] = spec["labels"]
    text_field: str = spec["text_field"]
    label_field: str = spec["label_field"]
    cleaning = spec.get("text_cleaning") or {}

    examples: list[Example] = []
    totals: Counter[str] = Counter()

    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text, counts = decode_escapes(
                str(row[text_field]),
                unescape_unicode=bool(cleaning.get("unescape_unicode", True)),
                unescape_quotes=bool(cleaning.get("unescape_quotes", True)),
            )
            totals.update(counts)

            index = int(row[label_field])
            if not 0 <= index < len(labels):
                raise ValueError(
                    f"{path.name}:{lineno}: label {row[label_field]!r} is outside the "
                    f"declared range 0..{len(labels) - 1}"
                )
            examples.append(Example(text=text, label=index, label_name=labels[index]))

    return examples, dict(totals)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_split_bundle(
    cfg: dict[str, Any],
    task: str,
    language: str,
    *,
    artifacts: str | Path | None = None,
    force: bool = False,
) -> SplitBundle:
    """Load the official splits for ``(task, language)`` straight from upstream.

    Asserts the sizes declared in configs/datasets.yaml, so a wrong file or a
    changed upstream fails here rather than producing a quietly different number.
    """
    if task not in _IMPLEMENTED_TASKS:
        raise NotImplementedError(
            f"no loader for task {task!r} yet; implemented: {_IMPLEMENTED_TASKS}. "
            "See Phase 7 in docs/deviations.md."
        )

    spec = get_dataset_spec(cfg, task)
    language_key, language_spec = resolve_language(cfg, language)
    files = fetch_split_files(spec, language_key, artifacts=artifacts, force=force)

    splits: dict[str, list[Example]] = {}
    cleaning_totals: Counter[str] = Counter()
    for split in SPLIT_NAMES:
        examples, counts = parse_split(files[split], spec)
        splits[split] = examples
        cleaning_totals.update(counts)

    bundle = SplitBundle(
        task=task,
        language=language_key,
        language_iso=language_spec["iso"],
        labels=spec["labels"],
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        source={
            "hf_id": spec["hf_id"],
            "hf_revision": spec.get("hf_revision"),
            "paper_ref": spec.get("paper_ref"),
            "cleaning": dict(cleaning_totals),
            "files": {name: str(path) for name, path in files.items()},
        },
    )
    assert_official_sizes(bundle, spec["expected_sizes"])
    return bundle


def save_split_bundle(bundle: SplitBundle, *, artifacts: str | Path | None = None) -> Path:
    """Write the three splits as JSONL plus a ``bundle.json`` describing them."""
    out = processed_dir(bundle.task, bundle.language_iso, artifacts)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        write_jsonl(out / f"{name}.jsonl", (ex.to_dict() for ex in bundle.split(name)))
    write_json(out / "bundle.json", bundle.meta())
    return out


def load_processed_bundle(
    cfg: dict[str, Any],
    task: str,
    language: str,
    *,
    artifacts: str | Path | None = None,
) -> SplitBundle | None:
    """Re-read a previously saved bundle. Returns ``None`` if it is not on disk.

    This is what makes later phases resumable: a run reads the cached splits
    instead of re-downloading and re-parsing, and still gets the size assertion.
    """
    spec = get_dataset_spec(cfg, task)
    language_key, language_spec = resolve_language(cfg, language)
    out = processed_dir(task, language_spec["iso"], artifacts)
    if not (out / "bundle.json").exists():
        return None

    meta = read_json(out / "bundle.json")
    bundle = SplitBundle(
        task=task,
        language=language_key,
        language_iso=language_spec["iso"],
        labels=meta["labels"],
        train=[Example.from_dict(r) for r in read_jsonl(out / "train.jsonl")],
        validation=[Example.from_dict(r) for r in read_jsonl(out / "validation.jsonl")],
        test=[Example.from_dict(r) for r in read_jsonl(out / "test.jsonl")],
        source=meta.get("source", {}),
    )
    assert_official_sizes(bundle, spec["expected_sizes"])
    return bundle
