"""End-to-end English SENTIMENT demo: the paper's experimental framework, no model downloads.

This is the single command to run for the demonstration. It:

1. loads the official English SENTIMENT splits from the local cache
   (run ``scripts/prepare_data.py`` once first - that is the only step that
   touches the network, and it downloads 3 small TSVs, not a model);
2. runs eight classifiers across the paper's four method families;
3. checks every training set against the held-out test split before use;
4. writes metrics, predictions, a CSV, figures and a markdown report to disk;
5. prints our results and the paper's reported results as two separate tables.

Nothing here downloads a pretrained model. The paper's five methods all require
one, so each is stood in for by a cheap substitute that fills the same slot in
the pipeline. Those substitutions are printed in the run banner, written into
``report.md`` under "Deviations", and listed in ``docs/deviations.md``. They are
NOT reruns of the paper's methods and are never presented as such.

Usage
-----
    python scripts/demo.py                       # English SENTIMENT, all methods
    python scripts/demo.py --force               # ignore cached results
    python scripts/demo.py --no-figures          # metrics only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcllm.config import (  # noqa: E402
    artifacts_dir,
    get_dataset_spec,
    load_datasets_config,
    resolve_language,
    set_seed,
)
from tcllm.data import (  # noqa: E402
    assert_no_test_leakage,
    load_processed_bundle,
    overlap_report,
)
from tcllm.eval import PAPER_ENGLISH_SENTIMENT, Result, evaluate, format_table  # noqa: E402
from tcllm.methods import (  # noqa: E402
    SeedWordZeroShot,
    generate_synthetic,
    make_few_shot_logreg,
    make_logreg,
    make_naive_bayes,
    make_svm,
)
from tcllm.utils.io import write_json, write_jsonl  # noqa: E402

RUN_NAME = "english_sentiment"

# The paper's synthetic SENTIMENT volume: 200 examples per category per LLM,
# across 3 LLMs. Our template generator matches the volume, not the source.
SYNTHETIC_PER_LABEL = 600

LEARNING_CURVE_SIZES = (10, 50, 100, 200, 500)

BANNER = """
================================================================================
 English SENTIMENT - the paper's experimental framework, run with no model
 downloads and no network access beyond the three dataset TSVs.
================================================================================
"""

DEVIATIONS_SHORT = """\
The paper's five methods each need a pretrained model (GTE 305M, mBERT 110M,
Qwen2.5-7B ~15 GB); this machine has no GPU and the models are not downloaded.
Each family is therefore exercised with a cheap substitute in the same slot:

  paper: zero-shot LLM prompting        -> seed-word lexicon (Liu et al. 2004)
  paper: FastFit few-shot fine-tuning   -> TF-IDF + LogReg on the same 10/label
  paper: LogReg on GTE embeddings       -> TF-IDF + LogReg (no embeddings)
  paper: LLM-generated synthetic data   -> template-and-slot generation, same volume
  paper: mBERT / Qwen2.5-7B fine-tuning -> not run (no local model)

The substitutions are the point of the demo: the pipeline, the splits, the
leakage guards, the metrics and the evaluation protocol are the paper's, so the
framework is verified end to end even though the model components are not.
"""


def slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    return "".join(keep).strip("_")[:60]


# --------------------------------------------------------------------------- #
# caching (project rule 9: resumable)
# --------------------------------------------------------------------------- #
def cached(key: str, *, runs: Path, force: bool = False) -> Result | None:
    path = runs / "results" / f"{key}.json"
    if force or not path.exists():
        return None
    from tcllm.utils.io import read_json

    data = read_json(path)
    data.pop("examples_per_second", None)
    return Result(**data)


def persist(key: str, result: Result, *, runs: Path, test_rows) -> None:
    write_json(runs / "results" / f"{key}.json", result.to_dict())
    write_jsonl(
        runs / "predictions" / f"{key}.jsonl",
        (
            {
                "text": row.text,
                "true_label": row.label_name,
                "predicted_label": result.labels[pred],
                "correct": int(row.label == pred),
            }
            for row, pred in zip(test_rows, result.predictions)
        ),
    )


def run_method(key, build, train, bundle, labels, *, runs, force, notes=""):
    """Evaluate one method, reusing the cached result when it is on disk."""
    done = cached(key, runs=runs, force=force)
    if done is not None:
        print(f"  [cached] {done.method:<44} acc={done.accuracy:.4f}")
        return done

    method = build()
    result = evaluate(method, train, bundle.final_eval(), labels, notes=notes)
    persist(key, result, runs=runs, test_rows=bundle.test)
    print(
        f"  [ ran  ] {result.method:<44} acc={result.accuracy:.4f} "
        f"f1={result.macro_f1:.4f} train={result.train_size} ({result.fit_seconds:.2f}s)"
    )
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def paper_vs_ours(by_key: dict[str, Result]) -> list[dict]:
    """Pair each of our four families with the paper's closest English number."""
    paper = PAPER_ENGLISH_SENTIMENT
    return [
        {
            "family": "Zero-shot\n(no training data)",
            "ours": by_key["zero_shot_lexicon"].accuracy,
            "paper": max(paper["GPT-4 (zero-shot)"], paper["Aya-Expanse-8B (zero-shot)"]),
        },
        {
            "family": "Few-shot\n(10 examples per class)",
            "ours": by_key["fewshot_10"].accuracy,
            "paper": paper["FastFit (few-shot, 10/label)"],
        },
        {
            "family": "Supervised\n(real training data)",
            "ours": by_key["full_logreg"].accuracy,
            "paper": max(paper["Logistic Reg. on GTE (real)"], paper["mBERT fine-tuned (real)"]),
        },
        {
            "family": "Supervised\n(synthetic training data)",
            "ours": by_key["synthetic_logreg"].accuracy,
            "paper": max(
                paper["Logistic Reg. on GTE (synthetic)"],
                paper["mBERT fine-tuned (synthetic)"],
                paper["Qwen2.5-7B SFT (synthetic)"],
            ),
        },
    ]


def write_csv(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "family", "train_size", "test_size", "accuracy", "macro_f1",
        "fit_seconds", "predict_seconds", "examples_per_second",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda r: -r.accuracy):
            row = result.to_dict()
            writer.writerow({name: row[name] for name in fields})


def write_report(path: Path, *, results, curve, bundle, figures, elapsed) -> None:
    ours = format_table(results)
    paper = "\n".join(
        f"{name:<44}{value:>8.3f}" for name, value in sorted(PAPER_ENGLISH_SENTIMENT.items())
    )
    figure_lines = "\n".join(f"- `{p.name}`" for p in figures) or "- (none)"

    path.write_text(
        f"""# English SENTIMENT - experimental results

*Generated by `scripts/demo.py`. Every number under "Results obtained here" was
measured on this machine; the numbers under "Reported by the authors" are
transcribed from arXiv:2502.11830 and are not our measurements.*

## Setup

| | |
|---|---|
| Task | SENTIMENT, 3 classes (negative / neutral / positive) |
| Language | English (`{bundle.language_iso}`) |
| Source | `{bundle.source.get('hf_id')}` @ `{bundle.source.get('hf_revision')}` |
| Splits used | train {bundle.sizes()['train']}, validation {bundle.sizes()['validation']}, test {bundle.sizes()['test']} (paper Table 1) |
| Test split | official, held out; never trained on, never used for selection |
| Wall clock | {elapsed:.1f}s total, CPU only |

## Results obtained here

```
{ours}
```

## Reported by the authors (arXiv:2502.11830, Table 6, English)

These are the paper's numbers for the same task and language. They are **not**
comparable row-for-row with the table above: the paper evaluates LLM prompting
and fine-tuned transformer models, this demo evaluates cheap stand-ins.

```
method                                       accuracy
{paper}
```

## Deviations

{DEVIATIONS_SHORT}
## Learning curve

Accuracy against labelled examples per class, drawn from the train split only:

```
{format_curve(curve)}
```

## Figures

{figure_lines}

## Reproducing

```
python scripts/prepare_data.py --task sentiment --language english
python scripts/demo.py
```
""",
        encoding="utf-8",
    )


def format_curve(curve) -> str:
    methods = []
    for row in curve:
        if row["method"] not in methods:
            methods.append(row["method"])
    sizes = sorted({row["n_per_label"] for row in curve}, key=lambda n: (n is None, n))
    header = f"{'examples/class':<16}" + "".join(f"{m[:22]:>24}" for m in methods)
    lines = [header, "-" * len(header)]
    for size in sizes:
        cells = []
        for method in methods:
            match = [r for r in curve if r["method"] == method and r["n_per_label"] == size]
            cells.append(f"{match[0]['accuracy']:.4f}" if match else "-")
        lines.append(f"{str(size):<16}" + "".join(f"{c:>24}" for c in cells))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="sentiment")
    parser.add_argument("--language", default="english")
    parser.add_argument("--config", default=None)
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="recompute every method")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    print(BANNER)
    started = time.perf_counter()
    set_seed(args.seed)

    cfg = load_datasets_config(args.config)
    spec = get_dataset_spec(cfg, args.task)
    language_key, language_spec = resolve_language(cfg, args.language)
    labels = list(spec["labels"])

    bundle = load_processed_bundle(cfg, args.task, args.language, artifacts=args.artifacts)
    if bundle is None:
        print(
            "ERROR: no processed splits on disk for "
            f"{args.task}/{language_key}.\n"
            "       Run this once first (it downloads three small TSVs, no model):\n"
            f"       python scripts/prepare_data.py --task {args.task} --language {language_key}",
            file=sys.stderr,
        )
        return 2

    runs = artifacts_dir(args.artifacts) / "runs" / RUN_NAME
    print(f"task        : {args.task}/{language_key}")
    print(f"splits      : {bundle.sizes()}  (official, per paper Table 1)")
    print(f"labels      : {labels}")
    print(f"output      : {runs}")
    print(f"seed        : {args.seed}")
    print(f"showing     : {DEVIATIONS_SHORT.strip().splitlines()[0]} ...\n")

    # ---- leakage guards, run before anything is trained -------------------- #
    print("leakage checks")
    overlap = overlap_report(bundle)
    print(f"  official train/test overlap       : {overlap['train_test_overlap']} rows")
    print(f"  official validation/test overlap  : {overlap['validation_test_overlap']} rows")
    print(f"  official train/validation overlap : {overlap['train_validation_overlap']} rows")

    by_key: dict[str, Result] = {}
    train_full = bundle.train
    test_rows = bundle.final_eval()

    # ---- supervised, full training data ----------------------------------- #
    print("\nfamily 1/4 - supervised on real data (stands in for GTE + LogReg / mBERT)")
    by_key["full_logreg"] = run_method(
        "full_logreg", make_logreg, train_full, bundle, labels, runs=runs, force=args.force,
        notes="full train split",
    )
    by_key["full_nb"] = run_method(
        "full_nb", make_naive_bayes, train_full, bundle, labels, runs=runs, force=args.force,
        notes="full train split",
    )
    by_key["full_svm"] = run_method(
        "full_svm", make_svm, train_full, bundle, labels, runs=runs, force=args.force,
        notes="full train split",
    )

    # ---- few-shot --------------------------------------------------------- #
    print("\nfamily 2/4 - few-shot (stands in for FastFit, same 10 examples per class)")
    for n_per_label, key in ((10, "fewshot_10"), (50, "fewshot_50"), (100, "fewshot_100")):
        sample = bundle.few_shot_sample(n_per_label=n_per_label, seed=args.seed)
        assert_no_test_leakage(sample, bundle, context=f"{key} few-shot sample")
        by_key[key] = run_method(
            key, make_few_shot_logreg, sample, bundle, labels, runs=runs, force=args.force,
            notes=f"{n_per_label} per class, train split only",
        )

    # ---- zero-shot -------------------------------------------------------- #
    print("\nfamily 3/4 - zero-shot (stands in for LLM prompting)")
    zero_shot = SeedWordZeroShot(labels)
    result = evaluate(zero_shot, [], test_rows, labels, notes="no training data at all")
    persist("zero_shot_lexicon", result, runs=runs, test_rows=bundle.test)
    by_key["zero_shot_lexicon"] = result
    print(f"  [ ran  ] {result.method:<44} acc={result.accuracy:.4f} f1={result.macro_f1:.4f} train=0")

    # ---- synthetic -------------------------------------------------------- #
    print("\nfamily 4/4 - trained on synthetic data only (stands in for LLM-generated data)")
    synthetic = generate_synthetic(labels, per_label=SYNTHETIC_PER_LABEL, seed=args.seed)
    assert_no_test_leakage(synthetic, bundle, context="synthetic corpus")
    write_jsonl(runs / "synthetic" / "train.jsonl", (ex.to_dict() for ex in synthetic))
    print(f"  generated {len(synthetic)} synthetic examples ({SYNTHETIC_PER_LABEL} per class), "
          "leakage check passed")
    by_key["synthetic_logreg"] = run_method(
        "synthetic_logreg", make_logreg, synthetic, bundle, labels, runs=runs, force=args.force,
        notes=f"{len(synthetic)} synthetic examples, no real data",
    )

    # ---- learning curve --------------------------------------------------- #
    print("\nlearning curve (accuracy vs labelled examples per class, train split only)")
    curve: list[dict] = []
    curve_builders = {
        "TF-IDF + LogReg": make_logreg,
        "TF-IDF + Naive Bayes": make_naive_bayes,
        "TF-IDF + Linear SVM": make_svm,
    }
    for method_name, build in curve_builders.items():
        for size in LEARNING_CURVE_SIZES:
            sample = bundle.few_shot_sample(n_per_label=size, seed=args.seed)
            assert_no_test_leakage(sample, bundle, context=f"curve {method_name} n={size}")
            key = f"curve_{slug(method_name)}_n{size}"
            point = run_method(key, build, sample, bundle, labels, runs=runs, force=args.force)
            curve.append({"method": method_name, "n_per_label": size,
                          "accuracy": point.accuracy, "macro_f1": point.macro_f1})
        full_key = f"curve_{slug(method_name)}_nfull"
        point = run_method(full_key, build, train_full, bundle, labels, runs=runs, force=args.force)
        curve.append({"method": method_name, "n_per_label": len(train_full),
                      "accuracy": point.accuracy, "macro_f1": point.macro_f1})

    # ---- persist summary -------------------------------------------------- #
    all_results = list(by_key.values())
    write_json(runs / "results.json", {
        "task": args.task,
        "language": language_key,
        "language_iso": bundle.language_iso,
        "seed": args.seed,
        "splits": bundle.sizes(),
        "fingerprints": bundle.fingerprints(),
        "synthetic_per_label": SYNTHETIC_PER_LABEL,
        "results": [r.to_dict() for r in all_results],
    })
    write_json(runs / "learning_curve.json", curve)
    write_json(runs / "paper_reference.json", {
        "source": "arXiv:2502.11830v1, Table 6 (English, SENTIMENT)",
        "note": "authors' reported accuracies; not produced by this code",
        "values": PAPER_ENGLISH_SENTIMENT,
    })
    write_csv(runs / "results.csv", all_results)

    # ---- print ------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("RESULTS OBTAINED HERE - English SENTIMENT, official test split")
    print("=" * 80)
    print(format_table(all_results))

    print("\n" + "=" * 80)
    print("REPORTED BY THE AUTHORS (arXiv:2502.11830, Table 6) - for reference only")
    print("=" * 80)
    for name, value in sorted(PAPER_ENGLISH_SENTIMENT.items(), key=lambda kv: -kv[1]):
        print(f"{name:<44}{value:>8.3f}")

    # ---- figures ---------------------------------------------------------- #
    figures: list[Path] = []
    if not args.no_figures:
        from tcllm import plotting

        figures_dir = runs / "figures"
        families = [by_key["zero_shot_lexicon"], by_key["fewshot_10"],
                    by_key["full_logreg"], by_key["synthetic_logreg"]]
        figures.append(plotting.plot_method_comparison(all_results, figures_dir / "fig1_method_comparison.png"))
        figures.append(plotting.plot_paper_vs_ours(paper_vs_ours(by_key), figures_dir / "fig2_paper_vs_ours.png"))
        figures.append(plotting.plot_learning_curve(curve, figures_dir / "fig3_learning_curve.png"))
        figures.append(plotting.plot_confusion_matrices(
            [{"title": r.method, "confusion": r.confusion, "accuracy": r.accuracy} for r in families],
            labels, figures_dir / "fig4_confusion_matrices.png",
        ))
        figures.append(plotting.plot_per_class_f1(
            [by_key["full_logreg"], by_key["fewshot_10"], by_key["zero_shot_lexicon"]],
            labels, figures_dir / "fig5_per_class_f1.png",
        ))
        print(f"\nwrote {len(figures)} figures to {figures_dir}")

    elapsed = time.perf_counter() - started
    write_report(runs / "report.md", results=all_results, curve=curve, bundle=bundle,
                 figures=figures, elapsed=elapsed)

    print(f"\noutputs under {runs}")
    print(f"  results.json, results.csv, learning_curve.json, report.md")
    print(f"  results/ and predictions/ (per method), synthetic/train.jsonl")
    print(f"\ntotal wall clock: {elapsed:.1f}s (CPU only, no model downloads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
