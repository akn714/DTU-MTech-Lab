# Demo notes — talking points for the faculty presentation

Written for: the presenter (you), to speak from in front of faculty.

---

## The one-paragraph version

I implemented the experimental framework of *"Text Classification in the LLM Era — Where do
we stand?"* (Vajjala & Shimangaud, arXiv:2502.11830). The framework is the paper's: its
official dataset splits, its four method families, its leakage discipline, its metric and
its evaluation protocol — running end to end on English sentiment classification. The paper's
five methods each require a pretrained model, and I did not download any, so each family is
exercised with a documented cheap substitute in the same slot. I am reporting what I measured,
and keeping the authors' published numbers in a separate table.

## Run it live (10 seconds)

```
cd "NLP/NLP Research Papers/Implementation"
python scripts/demo.py
```

If it is already cached it finishes in about 5 seconds; from scratch, about 10. It prints
our results and the paper's results as two clearly separated tables, then writes five figures
and a report to `artifacts/runs/english_sentiment/`.

First-time-only setup, if the data is not on disk yet (downloads three small TSVs, no model):

```
python scripts/prepare_data.py --task sentiment --language english
```

---

## What is genuinely the paper's (lead with this)

| | |
|---|---|
| Dataset | `cardiffnlp/tweet_sentiment_multilingual`, English |
| Splits | train 1839 / validation 324 / test 870 — **matches paper Table 1 exactly** |
| Labels | negative / neutral / positive |
| Test discipline | test split never trained on, never used to generate synthetic targets, never used for selection |
| Method families | zero-shot, few-shot, supervised on real data, supervised on synthetic data |
| Metric | accuracy on the held-out test split (macro-F1 recorded too) |

**The leakage discipline is structurally enforced, not promised.** The test split is only
reachable through a single `final_eval()` call; few-shot sets are drawn from the train split
by a method that cannot see test rows; and every training set — including the 1800 synthetic
examples — is re-checked against the test split at runtime before use. If that check ever
fails, the run aborts instead of reporting the number.

## What is substituted (say this before anyone asks)

The paper's five methods all need a pretrained model — GTE at 305M parameters, mBERT, and
Qwen2.5-7B at roughly 15 GB. This machine has no GPU, so each family runs with a substitute
in the same slot:

| Paper's method | Substitute I ran |
|---|---|
| Zero-shot LLM prompting | seed-word polarity lexicon (Liu et al., 2004) |
| FastFit few-shot fine-tuning | TF-IDF + logistic regression, same 10 examples per class |
| Logistic regression on GTE embeddings | TF-IDF + logistic regression, no embeddings |
| LLM-generated synthetic data | template generator, same volume (1800) |
| mBERT and Qwen2.5-7B fine-tuning | **not run** |

The substitutions do not change the framework. They change the instrument, which is exactly
why our numbers sit below the paper's.

---

## Results

```
TF-IDF + Naive Bayes                        0.502
Zero-shot seed lexicon                      0.502
TF-IDF + Logistic Regression                0.501
TF-IDF + Linear SVM                         0.488
Few-shot Logistic Regression (100/class)    0.448
TF-IDF + Logistic Regression (synthetic)    0.422
Few-shot Logistic Regression (50/class)     0.397
Few-shot Logistic Regression (10/class)     0.351
```

The paper's English results, for reference only: GPT-4 zero-shot 0.71, GTE + LogReg on real
data 0.65, FastFit few-shot 0.59, GTE + LogReg on synthetic data 0.59, Qwen2.5-7B SFT on real
data 0.58.

## Three findings worth presenting

**1 — The pipeline works, and the ordering is sane.** More labelled data helps monotonically:
0.35 at 10 examples per class, 0.45 at 100, 0.50 with the full training set. Training on
synthetic data alone (0.42) beats the 10-example few-shot setting but loses to real data
(0.50) — the same direction as the paper's own synthetic-data finding.

**2 — A lexicon's accuracy is misleading; its macro-F1 exposes it.** The zero-shot seed
lexicon ties for the best accuracy at 0.502 but has the worst macro-F1 of the top group
(0.466 vs 0.495). The confusion matrix shows why: it labels 75% of true negatives, 89% of
true neutrals and 55% of true positives as "neutral". On a three-class problem where neutral
is the majority, defaulting to neutral is a cheap way to buy accuracy. This is precisely why
macro-F1 is worth recording alongside accuracy — a point the paper itself raises (it reports
accuracy only, having found the two agreed in ordering).

**3 — Cheap models have a ceiling here.** Our best stand-in reaches about 0.50; the paper's
transformer and LLM methods reach 0.51–0.71. The gap is the value added by pretrained
representations and by instruction-following, measured on identical splits with an identical
protocol.

---

## The figures, in the order to show them

| Figure | What it shows | What to say |
|---|---|---|
| `fig1_method_comparison.png` | All eight of our methods, sorted, with the paper's range as a shaded band | "The band is where the paper's own English results land. We sit below it — which is what should happen, because we are not running their models." |
| `fig2_paper_vs_ours.png` | Our stand-in against the paper's closest number, per family | "Paired family by family. Blue is measured here, orange is what the authors published." |
| `fig3_learning_curve.png` | Accuracy against labelled examples per class | "Three classifiers, six training sizes. The curves are still rising at the full training set, so more data would keep helping." |
| `fig4_confusion_matrices.png` | Where each method goes wrong | "This is the most informative one. Look at the first panel — the lexicon puts almost everything in the neutral column." |
| `fig5_per_class_f1.png` | Per-class F1 for three methods | "Neutral is the hardest class for the trained models; the lexicon only looks competitive on it because it guesses neutral." |

All five are in `artifacts/runs/english_sentiment/figures/`.

---

## Questions you should expect

**"Did you actually implement the paper?"**
I implemented its experimental framework — the splits, the four method families, the leakage
guards, the metric, the evaluation protocol — and ran it end to end. I did not run the
paper's five models, because each needs a pretrained model download and this machine has no
GPU. Every substitution is documented in `docs/deviations.md` §6, with the file it lives in.

**"Are your numbers comparable to theirs?"**
Not row for row — different instruments. Comparable in kind and in protocol: same splits,
same task, same metric. The expected direction of the gap is downward, and that is what we see.

**"How do you know you did not accidentally train on the test set?"**
Structurally: the test rows are unreachable except through one function used only for final
evaluation, and every training set is re-checked against the test split at runtime. The
checks are in `src/tcllm/data/splits.py` and they abort the run on failure. On the English
splits the official train/test overlap is 0 anyway — though I found overlap in the French,
Hindi and Portuguese splits of the same official dataset, which is documented in §4.2.

**"Why is your zero-shot number as good as your supervised numbers?"**
Because on short tweets, explicit opinion words carry most of the signal, and because its
accuracy is inflated by collapsing everything into the majority class. Its macro-F1 is lower,
and the confusion matrix shows the collapse.

**"What would it take to close the gap?"**
GTE plus logistic regression and mBERT fine-tuning are the cheapest next steps — about
1.2 GB and 700 MB, both feasible on a free Colab T4. Zero-shot prompting via Ollama is the
heaviest at roughly 15 GB per model. Listed in order in `docs/deviations.md` §6.4.

**"What is not done?"**
Seven of the eight languages, and three of the four tasks. Those loaders raise
`NotImplementedError` rather than producing a plausible-looking number.

---

## Where everything lives

```
scripts/demo.py                     the single entry point
scripts/prepare_data.py             one-time data fetch (3 small TSVs, no model)
src/tcllm/data/                     loaders, splits, leakage guards
src/tcllm/methods/                  the four families (classical, zeroshot, synthetic)
src/tcllm/eval/                     metrics + the paper's numbers kept separate
src/tcllm/plotting/                 the five figures
configs/datasets.yaml               dataset ids, revisions, expected split sizes
docs/paper_summary.md               the paper's methodology and its tables
docs/deviations.md                  every deviation, including §6 for this demo
artifacts/runs/english_sentiment/   results.json, results.csv, predictions/, figures/, report.md
```
