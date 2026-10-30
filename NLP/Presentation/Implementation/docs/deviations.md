# Deviations from the paper

This project reproduces *Text Classification in the LLM Era — Where do we stand?*
(arXiv:2502.11830v1). Anything we do differently is recorded here, with the reason, so
that no reader has to guess why our numbers might differ from the authors'.

**Rule: paper-reported numbers and our numbers are never mixed.** The authors' figures live
in `docs/paper_summary.md` and (from Phase 8) `results/paper_reported/`. Ours live under
`results/ours/`. Nothing is ever copied from one to the other, and no result is estimated,
extrapolated or filled in from expectation.

---

## 1. Scope decisions

Agreed with the project owner on 2026-10-12, **revised on 2026-10-13**.

| Decision | Choice | Reason |
|---|---|---|
| LLM access | **Open-source models only. No GPT-4.** | No paid API budget. GPT-4 columns will be reported as *not run*, never as an estimate. |
| First target | **English SENTIMENT only**, all methods | Project rule 5. Smallest complete cell: it exercises the whole pipeline before anything is scaled. |
| Compute | Local CPU for light work + free Colab T4 for GPU work | Laptop is CPU-only (see §3). |

### 1.1 Revision, 2026-10-13: zero-download demo scope

The owner requires a working demonstration for a faculty presentation, with **no model
downloads of any size** and no scope beyond English SENTIMENT. The original plan
(8 languages x 4 tasks, GTE + mBERT + Qwen2.5-7B fine-tuning) is therefore **paused, not
abandoned** — see §6 for exactly what is and is not implemented.

The consequence is unavoidable and must be stated wherever these results appear: **every one
of the paper's five method families requires a pretrained model**, so none of them can be
reproduced without a download. What the demo reproduces is the *experimental framework* —
the official splits, the leakage discipline, the four method families, the metric and the
evaluation protocol — with cheap substitutes occupying the model slots.

## 2. Deviations

| # | Paper | This project | Reason / consequence |
|---|---|---|---|
| D1 | Zero-shot models: Qwen2.5-7B, Aya23-8B, Aya-Expanse-8B, **GPT-4** (`gpt-4-0613`) | Qwen2.5-7B, Aya-Expanse-8B, and Aya23-8B if it is still served by Ollama | No paid API. **The paper's strongest zero-shot baseline is absent**, so our zero-shot comparison is open-models-only. RQ1 can only be answered for open models. |
| D2 | Synthetic data from **three** sources (GPT-4, Qwen2.5-7B, Aya-Expanse-8B) | Two sources (Qwen2.5-7B, Aya-Expanse-8B) | Follows from D1. The paper's finding that *multi-source beats single-source* can still be tested with two sources, but not against GPT-4-inclusive data. |
| D3 | Qwen2.5-7B-Instruct **full** instruction fine-tuning on a V100 | QLoRA 4-bit on a free Colab T4 | A 7B full fine-tune needs roughly 60 GB of optimiser and gradient state; a T4 has 16 GB. Expect slightly different results from the paper's full SFT. |
| D4 | mBERT fine-tuned on Colab T4 / M1 Pro | Colab T4 where needed, local CPU otherwise | Matches the paper for the large datasets; local CPU is used only for the small ones. |
| D5 | Reported measure: **accuracy only** (macro-F1 computed but not reported, because the two always agreed in ordering) | **Both accuracy and macro-F1** are recorded | Project rule 10. An addition, not a contradiction — we can always report accuracy alone to match the paper. |
| D6 | Text used as distributed by the upstream dataset | Text is cleaned of two upstream escaping artefacts (§4.1) | The raw text contains literal `’` instead of `’`. Left uncleaned, every model sees corrupted text. |
| D7 | Dataset revision unstated | Pinned to `14b13edfbc4046892f6011d114c29c0f83170589` | Reproducibility. If upstream changed after the paper's run, our numbers differ slightly and for a known reason. |
| D8 | Not specified | SetFit will be run **in addition** for SENTIMENT (3 labels) | The paper dropped SetFit only above 10 categories, and does report it for small-label tasks in Appendix B. This is an addition, letting us check their claim locally. |
| D9 | Deferred | A **logistic regression** baseline is implemented first, before any LLM method | Project rule 5. It is the paper's own cheapest strong baseline for SENTIMENT. |

## 3. Hardware

| | Paper | This project |
|---|---|---|
| Primary | Apple M1 Pro, 32 GB | Windows 11 laptop, ~15 GB RAM, **no NVIDIA GPU** (PyTorch 2.11.0+cpu, `cuda.is_available() == False`) |
| GPU work | Colab T4 (mBERT), V100 (Qwen SFT) | Free Colab T4 |
| Local LLM serving | Ollama | Ollama — but on CPU this is ~1–2 min per generation, so bulk generation moves to Colab |

Consequence: the paper's runtime figures (FastFit 5–15 min, mBERT 0.5 h, Qwen 6 h + 3 h) are
not reproducible on this laptop and are not treated as targets. We record our own timings and
report both.

## 4. Data issues found during Phase 1

These are properties of the **upstream** data, not of our code. They are preserved rather
than silently corrected, per project rule 4 (preserve the official splits).

### 4.1 Double-escaped Unicode in the source files

`cardiffnlp/tweet_sentiment_multilingual` ships JSONL containing `\\u2019`. A single
`json.loads` therefore yields the six literal characters `’` rather than the character
`’`. Any pipeline reading this dataset without cleaning gets corrupted text.

Measured in English train: **225 of 1839 rows**, 386 escape sequences.

We decode exactly two things, and count both in the audit:

- `\uXXXX` → the character it denotes (`’` → `’`, `,` → `,`)
- `\"` → `"`

Everything else is preserved **on purpose**. The same dataset contains genuine kaomoji such
as `\(^-^ )/` and `\m/`, which a blanket unescape would destroy. The audit reports how many
backslashes were left untouched so this choice stays visible.

### 4.2 Train/test overlap in the official splits

Detected by `scripts/prepare_data.py`, which reports duplicates and overlaps for every
language. Results at the pinned revision:

| Language | train ↔ validation | train ↔ **test** | validation ↔ **test** |
|---|---|---|---|
| Arabic | 0 | 0 | 0 |
| **English** | 0 | 0 | 0 |
| French | 0 | **6** | **1** |
| German | 0 | 0 | 0 |
| Hindi | 0 | **2** | 0 |
| Italian | 0 | 0 | 0 |
| Portuguese | 1 | **1** | 0 |
| Spanish | 0 | 0 | 0 |

English — our Phase 1 target — is clean. For French, Hindi and Portuguese a handful of test
rows also appear in train, which slightly inflates those languages' scores for every method
including the paper's. The overlap is small (at most 6 of 870, 0.7%) and we do **not** repair
it: removing rows would break the official split, which rule 4 forbids. It is recorded here
and surfaced in each language's `audit.json` so it can be mentioned in the final report.

### 4.3 Items to verify before Phase 7

- The paper's INTENT and SCENARIO train size is 12000; the public MASSIVE 1.0 train split is
  11514. The exact upstream revision matters and will be pinned and checked when those tasks
  are added.
- TAXI1500's HuggingFace distribution needs identifying and pinning (Ma et al., 2023).

## 5. Official code and datasets

The paper states that all implementation code and the generated synthetic datasets were
"provided as supplementary material", and that the datasets used are publicly available.

### 5.1 The authors' code is not publicly available

Searched on 2026-10-13. Findings:

- The **arXiv abstract page has no ancillary-files section** and contains no code links; the
  submission is marked *Pre-print*. The arXiv source bundle is the paper's LaTeX only.
- The **first author's GitHub account** (`nishkalavallabhi`, confirmed to be Sowmya Vajjala by
  her `OneStopEnglishCorpus`, `SOTANER` and `practicalnlp` repositories) has 20 public
  repositories. **None relates to this paper** — the closest in time are unrelated work from
  2025–2026.
- **GitHub repository search** for the paper title, its arXiv id, and the author names returns
  no repository by the authors. (The search endpoint was verified working against control
  queries before this conclusion was drawn.)
- The one name-matching third-party repository (`Yousef9Amr/Text-Classification-in-the-era-of-LLM`)
  is a single notebook, 0 stars, with a one-line README, and it was created a week *before* the
  paper appeared on arXiv. It is not authoritative and is **not** used.
- ACL Anthology has no entry for the paper.

**Conclusion:** the supplementary material was not released publicly, so project rule 2
("prefer the authors' implementation when accessible") does not apply — it is not accessible.
Everything here is implemented from the paper's own description. This is recorded as an open
fact, not an assumption: if a repository appears later, it should be re-evaluated.

### 5.2 Official implementations of the *methods* the paper uses

These are authoritative and will be preferred over hand-rolled equivalents where a method is
reimplemented:

| Component | Source | Status |
|---|---|---|
| FastFit (the paper's few-shot method) | `IBM/fastfit` — Yehudai & Bandel (2024) | available |
| SetFit (paper's Appendix B comparison) | `huggingface/setfit` | available |
| Structured zero-shot output | Instructor (`python.useinstructor.com`) | available; the paper's own choice |
| Local open-LLM serving | Ollama | available; the paper's own choice |

### 5.3 Datasets

**Confirmed available and matching.** `cardiffnlp/tweet_sentiment_multilingual` is public,
contains all 8 languages with the official splits, and matches Table 1 exactly
(1839 / 324 / 870 for every language) at the revision pinned in `configs/datasets.yaml`.

## 6. Zero-download demo scope (2026-10-13)

This section is the authoritative record of what the demonstration actually runs. If a
number from this project is quoted anywhere, the substitution table below applies to it.

### 6.1 Substitutions

Each of the paper's five method families is exercised by a substitute that fills the same
*slot in the pipeline* — same splits, same leakage guards, same metric, same evaluation
protocol — while using no pretrained model. The substitutes are **not** the paper's methods.

| # | Paper's method | Model it needs | Substitute implemented | Where |
|---|---|---|---|---|
| S1 | Zero-shot LLM prompting | Qwen2.5-7B / Aya-Expanse-8B (~15 GB) | Seed-word polarity lexicon with negation handling (Liu et al., 2004 — the pre-LLM zero-shot method the paper itself cites) | `src/tcllm/methods/zeroshot.py` |
| S2 | FastFit few-shot fine-tuning | `paraphrase-multilingual-mpnet-base-v2` (~1.1 GB) | TF-IDF + logistic regression on the **same 10 examples per class** | `src/tcllm/methods/classical.py` |
| S3 | Logistic regression on GTE embeddings | `gte-multilingual-base` (~1.2 GB) | TF-IDF + logistic regression (no embeddings) | `src/tcllm/methods/classical.py` |
| S4 | Classifiers trained on LLM-generated synthetic data | GPT-4 / Qwen2.5-7B / Aya-Expanse-8B | Template-and-slot generator, **volume-matched** at 1800 examples (600 per class = the paper's 200/cat x 3 LLMs) | `src/tcllm/methods/synthetic.py` |
| S5 | Fine-tuned mBERT | mBERT (~700 MB) | **Not run.** | — |
| S6 | Qwen2.5-7B SFT | Qwen2.5-7B (~15 GB) | **Not run.** | — |

S1 is a genuine zero-shot classifier — it never sees a labelled example, and `fit()` raises
if it is given one, so the comparison cannot silently become supervised. It is simply a much
poorer opinion model than a 7B language model, which is the expected direction of the gap.

### 6.2 What is genuinely the paper's, and what is not

**Preserved exactly:**

- The official train/validation/test splits from `cardiffnlp/tweet_sentiment_multilingual`
  at the pinned revision — 1839 / 324 / 870, matching paper Table 1 for English.
- The test split is never trained on, never used to generate synthetic targets, and never
  used for selection. Few-shot sets are drawn from the train split only, enforced
  structurally by `SplitBundle` and re-checked at runtime by `assert_no_test_leakage`.
- The four method families, the task, the language and the label set.
- Accuracy on the official test split, with macro-F1 recorded alongside (deviation D5).
- Training and inference time, per method (project rule 10).

**Substituted or absent:** every pretrained model component (§6.1).

**Scale:** 8 languages x 4 tasks becomes **1 language x 1 task**. The other three tasks have
no loader; `loaders.py` raises `NotImplementedError` rather than pretending otherwise.

### 6.3 Consequences for interpreting the results

1. **Our numbers are not comparable row-for-row with the paper's.** A TF-IDF model and a
   fine-tuned transformer are different instruments. The figures draw the paper's English
   results as muted dashed *reference guides* and keep our measured bars in a separate
   colour, so the two are never read as one series.
2. **The expected direction of the gap is downward**, and that is what the run shows: our
   best stand-in reaches about 0.50 where the paper's models reach 0.51-0.71. A substitute
   that *matched* the paper would be the surprising result, and would suggest a leak.
3. **Our zero-shot result is unusually strong relative to our supervised ones**, because
   short tweets are dominated by explicit opinion words and a lexicon does well on them.
   This is a property of the substitute, not evidence that zero-shot beats supervised
   learning. The per-class figure shows why its headline accuracy is misleading: it assigns
   "neutral" to 75%/89%/55% of each true class, so its macro-F1 is well below its accuracy.
4. **Synthetic-only training loses to real-data training here** (0.42 vs 0.50), consistent
   in direction with the paper's own finding that synthetic data is competitive but not
   equal to real data.
5. **No result in this project was fabricated, estimated or transcribed.** The paper's
   numbers live in `src/tcllm/eval/metrics.py` under `PAPER_ENGLISH_SENTIMENT` and are
   emitted to a separate file (`paper_reference.json`) and a separate table.

### 6.4 What would be needed to close the gap

In rough order of cost, to move from the demo to the paper's actual methods:

1. GTE + logistic regression and mBERT fine-tuning — ~1.2 GB and ~700 MB, both feasible on
   a free Colab T4, and the smallest gap to close (S3, S5).
2. FastFit on `paraphrase-multilingual-mpnet-base-v2` — ~1.1 GB, CPU-feasible at this
   dataset size (S2).
3. Zero-shot prompting via Ollama — the heaviest, ~15 GB per model, and the largest
   substitution (S1).
