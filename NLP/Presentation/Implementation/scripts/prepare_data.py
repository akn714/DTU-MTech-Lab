#!/usr/bin/env python
"""Phase 1 — materialise the official splits for one (task, language) pair.

What this does
--------------
1. Downloads the upstream files at the revision pinned in ``configs/datasets.yaml``
   into ``artifacts/data/raw/hf_cache``. After the first run this is offline and
   byte-identical.
2. Cleans the known upstream escaping artefacts (see ``docs/deviations.md``).
3. Asserts the loaded sizes against the official sizes in the config, which come
   from Table 1 of the paper.
4. Writes ``artifacts/data/processed/<task>/<iso>/{train,validation,test}.jsonl``
   plus ``audit.json``.

Why an audit file
-----------------
Every later result records the fingerprint of the splits it used. The audit is
what lets a table produced weeks from now be traced back to exactly which rows
produced it — and it is where duplicate and test-overlap counts are surfaced
rather than assumed away.

How to run
----------
    python scripts/prepare_data.py --task sentiment --language english
    python scripts/prepare_data.py --task sentiment --all-languages
    python scripts/prepare_data.py --task sentiment --language english --force

What to expect
--------------
Sizes 1839 / 324 / 870 for every language, perfectly balanced across the three
sentiment classes, and zero train/test or validation/test overlap. Exits non-zero
on a size mismatch or (with ``--strict``) on any warning.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcllm.config import (  # noqa: E402
    artifacts_dir,
    get_dataset_spec,
    load_datasets_config,
    processed_dir,
    resolve_language,
    set_seed,
)
from tcllm.data import (  # noqa: E402
    SPLIT_NAMES,
    SplitBundle,
    load_processed_bundle,
    load_split_bundle,
    overlap_report,
    save_split_bundle,
)
from tcllm.utils.io import write_json  # noqa: E402

RULE = "=" * 74


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _summary(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 1),
        "median": statistics.median(values),
        "max": max(values),
    }


def text_stats(bundle: SplitBundle) -> dict[str, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for name in SPLIT_NAMES:
        examples = bundle.split(name)
        if not examples:
            continue
        stats[name] = {
            "chars": _summary([len(ex.text) for ex in examples]),
            "words": _summary([len(ex.text.split()) for ex in examples]),
        }
    return stats


def build_audit(bundle: SplitBundle, spec: dict) -> tuple[dict, list[str]]:
    """Assemble the audit record and the list of warnings it produced."""
    expected = spec["expected_sizes"]
    actual = bundle.sizes()
    overlap = overlap_report(bundle)
    warnings: list[str] = []

    for name in expected:
        if name in actual and actual[name] != expected[name]:
            warnings.append(f"size mismatch for {name}: expected {expected[name]}, got {actual[name]}")

    for name in ("train_test_overlap", "validation_test_overlap"):
        if overlap.get(name):
            warnings.append(
                f"{name.replace('_', ' ')} is {overlap[name]} - test rows also appear in "
                "another split, which would inflate the reported score"
            )

    for name in SPLIT_NAMES:
        for label, count in bundle.label_distribution(name).items():
            if count == 0:
                warnings.append(f"split {name} has no examples for label {label!r}")

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": bundle.task,
        "language": bundle.language,
        "language_iso": bundle.language_iso,
        "hf_id": bundle.source.get("hf_id"),
        "hf_revision": bundle.source.get("hf_revision"),
        "paper_ref": bundle.source.get("paper_ref"),
        "sizes": actual,
        "expected_sizes": dict(expected),
        "sizes_match_official": all(actual.get(k) == v for k, v in expected.items()),
        "labels": list(bundle.labels),
        "label_distribution": {name: bundle.label_distribution(name) for name in SPLIT_NAMES},
        "text_stats": text_stats(bundle),
        "overlap": overlap,
        "cleaning": bundle.source.get("cleaning", {}),
        "fingerprints": bundle.fingerprints(),
        "warnings": warnings,
    }
    return audit, warnings


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_audit(audit: dict, cached: bool) -> None:
    spec_lang = audit["language"]
    print(RULE)
    print(f"{audit['task'].upper()} / {spec_lang} ({audit['language_iso']})" + ("   [cached]" if cached else ""))
    print(RULE)
    print(f"source        {audit['hf_id']} @ {(audit['hf_revision'] or 'main')[:12]}")
    print(f"paper ref     {audit['paper_ref']}")

    sizes = audit["sizes"]
    expected = audit["expected_sizes"]
    print(
        "splits        "
        + "  ".join(f"{name}={sizes.get(name, 0)}" for name in SPLIT_NAMES)
    )
    print(
        "expected      "
        + "  ".join(f"{name}={expected.get(name, '-')}" for name in SPLIT_NAMES)
        + ("   OK" if audit["sizes_match_official"] else "   MISMATCH")
    )
    print(f"labels ({len(audit['labels'])})    {', '.join(audit['labels'])}")

    print("\nlabel balance")
    header = f"  {'split':<11}" + "".join(f"{label:>11}" for label in audit["labels"])
    print(header)
    for name in SPLIT_NAMES:
        dist = audit["label_distribution"][name]
        print(f"  {name:<11}" + "".join(f"{dist.get(label, 0):>11}" for label in audit["labels"]))

    print("\ntext length (characters)")
    print(f"  {'split':<11}{'min':>7}{'mean':>9}{'median':>9}{'max':>7}")
    for name, stats in audit["text_stats"].items():
        c = stats["chars"]
        print(f"  {name:<11}{c['min']:>7}{c['mean']:>9}{c['median']:>9}{c['max']:>7}")

    overlap = audit["overlap"]
    print("\nduplicates / overlap")
    print(
        "  within-split duplicates   "
        + "  ".join(f"{name}={overlap.get(f'{name}_duplicates', 0)}" for name in SPLIT_NAMES)
    )
    print(f"  train <-> validation      {overlap.get('train_validation_overlap', 0)}")
    print(f"  train <-> test            {overlap.get('train_test_overlap', 0)}   <- must be 0")
    print(f"  validation <-> test       {overlap.get('validation_test_overlap', 0)}   <- must be 0")

    cleaning = audit["cleaning"]
    if cleaning:
        print("\ntext cleaning (upstream escaping artefacts)")
        print(f"  unicode escapes decoded (\\uXXXX -> char)   {cleaning.get('unicode_escapes', 0)}")
        print(f"  escaped quotes decoded   (\\\" -> \")        {cleaning.get('escaped_quotes', 0)}")
        print(f"  backslashes preserved (genuine, e.g. \\m/)   {cleaning.get('backslashes_preserved', 0)}")

    print("\nfingerprints (sha256, order-independent)")
    for name, value in audit["fingerprints"].items():
        print(f"  {name:<11} {value[:32]}")

    if audit["warnings"]:
        print("\nWARNINGS")
        for warning in audit["warnings"]:
            print(f"  ! {warning}")
    else:
        print("\nno warnings")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def prepare_one(
    cfg: dict,
    task: str,
    language: str,
    *,
    artifacts: str | Path | None,
    force: bool,
    quiet: bool,
) -> tuple[dict, list[str]]:
    spec = get_dataset_spec(cfg, task)
    language_key, language_spec = resolve_language(cfg, language)

    bundle = None if force else load_processed_bundle(cfg, task, language_key, artifacts=artifacts)
    cached = bundle is not None

    if bundle is None:
        bundle = load_split_bundle(cfg, task, language_key, artifacts=artifacts, force=force)
        out = save_split_bundle(bundle, artifacts=artifacts)
    else:
        out = processed_dir(task, language_spec["iso"], artifacts)

    audit, warnings = build_audit(bundle, spec)
    audit["processed_dir"] = str(out)
    write_json(out / "audit.json", audit)

    if not quiet:
        print_audit(audit, cached)
    return audit, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialise the official splits for a task/language pair (Phase 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="sentiment", help="task key from configs/datasets.yaml")
    parser.add_argument("--language", default="english", help="language key (english) or ISO code (en)")
    parser.add_argument("--all-languages", action="store_true", help="process every paper language")
    parser.add_argument("--config", default=None, help="path to datasets.yaml")
    parser.add_argument("--artifacts", default=None, help="artifacts root (default: <project>/artifacts)")
    parser.add_argument("--force", action="store_true", help="re-download and re-parse, ignoring the cache")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any warning is raised")
    parser.add_argument("--seed", type=int, default=42, help="global seed (recorded for provenance)")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_datasets_config(args.config)
    root = artifacts_dir(args.artifacts)
    print(f"artifacts root: {root}\n")

    if args.all_languages:
        languages = [resolve_language(cfg, lang)[0] for lang in cfg.get("paper_languages", [])]
    else:
        languages = [resolve_language(cfg, args.language)[0]]

    all_warnings: dict[str, list[str]] = {}
    for index, language in enumerate(languages):
        if index:
            print()
        try:
            _, warnings = prepare_one(
                cfg, args.task, language, artifacts=args.artifacts, force=args.force, quiet=False
            )
        except Exception as exc:  # noqa: BLE001 - a CLI should report, not traceback
            print(f"\nFAILED for {language}: {type(exc).__name__}: {exc}")
            return 1
        all_warnings[language] = warnings

    if args.all_languages:
        print(f"\n{RULE}\nSUMMARY across {len(languages)} languages\n{RULE}")
        for language, warnings in all_warnings.items():
            status = "ok" if not warnings else f"{len(warnings)} warning(s)"
            print(f"  {language:<12} {status}")

    if args.strict and any(all_warnings.values()):
        print("\n--strict: failing because warnings were raised")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
