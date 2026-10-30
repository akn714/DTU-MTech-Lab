# Text Classification in the LLM Era — implementation

A reproducible implementation of **Vajjala & Shimangaud (2025), _Text Classification in the
LLM Era — Where do we stand?_**, arXiv:[2502.11830](https://arxiv.org/abs/2502.11830).

The paper compares zero-shot LLM classification, few-shot fine-tuning, synthetic-data-based
classifiers and full-data supervised classifiers across 4 tasks and 8 languages. See
[docs/paper_summary.md](docs/paper_summary.md) for the methodology and the authors' reported
results.

> **Paper numbers and our numbers are never mixed.** The authors' results are transcribed in
> [docs/paper_summary.md](docs/paper_summary.md); ours are regenerated into `results/ours/`.
> Everything we do differently is itemised in [docs/deviations.md](docs/deviations.md).

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Paper summary, plan, deviations log | done |
| 1 | Project skeleton, config, data loading, split guards | **done** |
| 1b | **Zero-download English SENTIMENT demo** — 4 method families, 8 classifiers, 5 figures | **done** (2026-10-13) |
| 2 | Logistic regression on GTE embeddings — English SENTIMENT | paused (needs a ~1.2 GB model) |
| 3 | Zero-shot open LLM (Colab) | planned |
| 4 | FastFit few-shot (10 per label) | planned |
| 5 | Synthetic data generation + classifier | planned |
| 6 | mBERT fine-tune, Qwen QLoRA-SFT (Colab) | planned |
| 7 | Scale to 8 languages x 4 tasks | planned |
| 8 | Comparison tables and figures | done for the demo |

Planned approach, for context: open-source LLMs only (no GPT-4, no paid API), starting with
English sentiment and expanding. Free Colab T4 for GPU work; this laptop is CPU-only.

Phase 1b is a **demonstration scope**: the paper's experimental framework — official splits,
four method families, leakage guards, metrics, evaluation protocol — run end to end with no
model downloads. Every one of the paper's five methods needs a pretrained model, so each
family is exercised by a documented cheap substitute in the same slot. Read
[docs/deviations.md](docs/deviations.md) §6 before quoting any number from it, and
[docs/demo_notes.md](docs/demo_notes.md) for the presentation walkthrough.

## Demo (English SENTIMENT, no model downloads)

```bash
# once: fetch the three dataset TSVs (no model is downloaded)
python scripts/prepare_data.py --task sentiment --language english

# then: run every method, print the tables, write figures and a report
python scripts/demo.py
```

Finishes in about 10 seconds on CPU and writes to `artifacts/runs/english_sentiment/`:
`results.json`, `results.csv`, `learning_curve.json`, `report.md`, per-method files under
`results/` and `predictions/`, the generated corpus under `synthetic/`, and five figures
under `figures/`. Re-running is cheap — cached per-method results are reused until
`--force` is passed.

The run prints our measured results and the paper's published results as **two separate
tables**; the two are never merged.


## Requirements

Python 3.10+ (developed on 3.13.7). Two dependency files, for two different machines.

### Windows / local (CPU)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional, to make `import tcllm` work from anywhere instead of relying on the scripts'
own path bootstrap:

```bash
python -m pip install -e .
```

### Google Colab (GPU)

Colab already ships a CUDA build of PyTorch. **Never install `requirements.txt` on Colab** —
its pinned CPU torch would replace the CUDA wheel and every GPU run would silently fall back
to CPU.

```python
!git clone <your-repo-url> tcllm-project
%cd tcllm-project/NLP/"NLP Research Papers"/Implementation
!pip install -q -r requirements-colab.txt -r requirements.txt
```

If the two files conflict over torch, install the Colab file alone and add the core packages
individually — Colab's preinstalled torch must win.

## Quick start

Phase 1 downloads the official splits, cleans them, verifies them against the sizes in
Table 1 of the paper, and writes an audit.

```bash
# one language (the Phase 1 target)
python scripts/prepare_data.py --task sentiment --language english

# all eight languages
python scripts/prepare_data.py --task sentiment --all-languages

# re-download from scratch, ignoring the cache
python scripts/prepare_data.py --task sentiment --language english --force

# fail the run if any warning is raised (useful in CI)
python scripts/prepare_data.py --task sentiment --all-languages --strict
```

Expected output, for English: `train=1839  validation=324  test=870`, matching the paper's
Table 1, perfectly balanced at 613/613/613, with **no** train/test or validation/test overlap.
A cached re-run prints `[cached]` and does no network work.

## What Phase 1 produced

```
artifacts/
├── data/
│   ├── raw/hf_cache/                         # pinned upstream downloads
│   └── processed/sentiment/en/
│       ├── train.jsonl   (1839 rows)
│       ├── validation.jsonl  (324 rows)
│       ├── test.jsonl    (870 rows)
│       ├── bundle.json   # labels, sizes, per-split fingerprints, provenance
│       └── audit.json    # sizes, label balance, text stats, overlaps, cleaning counts
```

Each row is `{"text": ..., "label": 0|1|2, "label_name": "negative"|"neutral"|"positive"}`.
The six other languages are written the same way under `processed/sentiment/<iso>/`
(`ar, fr, de, hi, it, pt, es`).

`bundle.json` and `audit.json` carry a **fingerprint** (sha256, order-independent) of each
split. Later phases record the fingerprint they used, so any table can be traced back to
exactly which rows produced it.

## Layout

```
Implementation/
├── configs/datasets.yaml      # task/language registry + official split sizes
├── src/tcllm/
│   ├── config.py              # YAML loading, paths, seeding
│   ├── data/
│   │   ├── loaders.py         # download, clean, parse, cache
│   │   └── splits.py          # SplitBundle + the leakage guards
│   ├── methods/
│   │   ├── base.py            # the Method protocol every classifier satisfies
│   │   ├── classical.py       # TF-IDF + LogReg / Naive Bayes / Linear SVM
│   │   ├── zeroshot.py        # seed-word lexicon, refuses to be fitted
│   │   └── synthetic.py       # template-and-slot corpus generator
│   ├── eval/metrics.py        # accuracy, macro-F1, timing; the paper's numbers kept apart
│   ├── plotting/plots.py      # the five report figures
│   └── utils/io.py            # atomic writes, JSONL, fingerprints
├── scripts/                   # one CLI per pipeline step (prepare_data, demo)
├── docs/                      # paper summary, deviations log, demo notes
└── artifacts/                 # generated; gitignored
```

**The leakage rule is structural, not a convention.** `SplitBundle.few_shot_sample()` and
`train_examples_for_label()` read `self.train` directly — there is no code path that can hand
a test row to a learner. Test rows are reachable only through `SplitBundle.final_eval()`,
whose name states the intent at every call site. `assert_no_test_leakage()` catches the
remaining case, where generated rows are traced back to the test split.

## Configuration

`configs/datasets.yaml` is the single source of truth for dataset metadata: HuggingFace id,
**pinned revision**, file layout, label names and the official split sizes. Those sizes are
asserted at load time, so a changed upstream or a wrong path fails loudly instead of quietly
producing a different experiment.

Redirect the artifacts root — useful on Colab, where Drive survives a disconnect but is slow,
and local disk is fast but ephemeral:

```bash
export TCLLM_ARTIFACTS=/content/drive/MyDrive/tcllm-artifacts   # bash
$env:TCLLM_ARTIFACTS = "D:\tcllm-artifacts"                     # PowerShell
```

or pass `--artifacts <path>`.

## Known upstream data issues

Two defects in the source data are handled explicitly and documented in full in
[docs/deviations.md](docs/deviations.md):

1. **Double-escaped Unicode.** The JSONL contains `\\u2019`, so a plain `json.loads` gives
   the six characters `\u2019` instead of `’` (225 of 1839 English train rows). We decode
   `\uXXXX` and `\"`, and deliberately preserve every other backslash, because the data
   contains real kaomoji such as `\(^-^ )/` and `\m/`.
2. **Train/test overlap** in three languages' official splits (French 6, Hindi 2,
   Portuguese 1). English is clean. We report it and leave the official splits intact.

## Reproducibility notes

- The upstream revision is pinned, so runs are byte-identical over time.
- All writes under `artifacts/` are atomic, so an interrupted run never leaves a
  half-written file that the resume logic would wrongly trust.
- Re-running a completed step is a no-op that re-verifies the cached data.
- `--seed` is recorded for provenance from Phase 2 onward.
