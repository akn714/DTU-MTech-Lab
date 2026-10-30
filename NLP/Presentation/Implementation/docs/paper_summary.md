# Paper summary — experimental methodology

**Vajjala, S. & Shimangaud, S. (2025). _Text Classification in the LLM Era — Where do we stand?_**
arXiv:2502.11830v1 [cs.CL], 17 Feb 2025.

Read in full (14 pages, including Appendix A "Details about the Experimental Settings"
and Appendix B "Additional Results") before any code was written, per project rule 1.

---

## 1. Research questions

| # | Question |
|---|---|
| RQ1 | How well does zero-shot prompting of LLMs (open and proprietary) fare compared to building classifiers with full training data? |
| RQ2 | Does few-shot fine-tuning offer any benefits over zero-shot classification? |
| RQ3 | How well does a synthetic-data-based classifier fare compared to zero-shot classification with LLMs? |
| RQ4 | Is supervised/instruction fine-tuning of LLMs the way to go for text classification? |

## 2. Datasets

Four publicly available datasets, each with 8 language subsets = **32 datasets**. All use
the **official train/validation/test splits**. Arabic and Hindi are in native script; the
other six languages are Roman script.

| Dataset | Task type | Labels | Train | Test | Source |
|---|---|---|---|---|---|
| SENTIMENT | sentiment (subjective) | 3 | 1839 | 870 | Multilingual Twitter sentiment (Barbieri et al., 2022) |
| TAXI1500 | topic | 6 | 860 | 111 | Taxi1500 (Ma et al., 2023) — Bible verses |
| SCENARIO | topic | 18 | 12000 | 2974 | MASSIVE `scenario` field (FitzGerald et al., 2023) |
| INTENT | intent | 60 | 12000 | 2974 | MASSIVE `intent` field (FitzGerald et al., 2023) |

Languages: Arabic, English, French, German, Hindi, Italian, Portuguese, Spanish.
TAXI1500 labels: `recommendation, faith, description, sin, grace, violence`.

MASSIVE is a single dataset; the authors model its `intent` and `scenario` fields as two
separate tasks. Note the paper's INTENT/SCENARIO train size of 12000 is worth verifying
against the upstream release during Phase 7 — the public MASSIVE 1.0 train split is 11514.

## 3. Methods

### 3.1 Zero-shot prompting
- Models: **Qwen2.5-7B**, **Aya23-8B**, **Aya-Expanse-8B** (open, run locally with Ollama),
  and **GPT-4** (`gpt-4-0613`, proprietary, via API).
- Output structure controlled with **Instructor** + **Pydantic**; the label list is passed
  as `catnames`.
- **All prompts are in English.** Prompting in the target language gave worse results in
  their early experiments.
- Models still emitted explanations beyond the label (up to 10% for some open models).
  **Those responses were counted as classification errors.**
- Few-shot *prompting* was not attempted, because of the large label sets.

### 3.2 Few-shot fine-tuning
- **FastFit** (Yehudai & Bandel, 2024) — batch contrastive learning plus a token-similarity
  score.
- **10 examples per label**, the best setting in the original FastFit paper.
- Base model: `paraphrase-multilingual-mpnet-base-v2`.
- **SetFit** was also tried but became intractable above 10 categories, so only FastFit is
  reported for the large-label tasks; SetFit appears in Appendix B for the small ones.

### 3.3 Synthetic data generation
- **Three sources**: GPT-4, Qwen2.5-7B, Aya-Expanse-8B — **equal amounts from each**, to
  force diversity. They found multi-source better than any single source.
- One fixed English prompt template for all models and languages (Appendix A of the paper):

  > generate a **{text_lang}** language text that looks like **{text_genre}** that can be
  > categorized as **{list_of_cats[i]}** in the context of **{task_desc}**. The generated
  > text should be under 50 words, and ensure some diversity of vocabulary in the generated
  > texts.

- Generated sizes, per language:

  | Dataset | # train | Description |
  |---|---|---|
  | SENTIMENT | 1800 | 200 per category, per LLM |
  | TAXI1500 | 1800 | 100 per category, per LLM |
  | INTENT | 13500 | 75 per category, per LLM |
  | SCENARIO | 13500 | taken from the INTENT dataset |

### 3.4 Classifiers trained on real and synthetic data
Three classifiers, each trained twice — once on real data only, once on synthetic only:

1. **Logistic regression** over `gte-multilingual-base` (305M params) embeddings, no fine-tuning.
2. **mBERT** (`google-bert/bert-base-multilingual-cased`) fine-tuned, 5 epochs.
3. **Qwen2.5-7B-Instruct** instruction fine-tuned, 3 epochs (10 for TAXI1500).

Instruction format for Qwen SFT:

```
<s>[INST] Consider the text: "{input_text}" Please select the most relevant category for
the given text from following OPTIONS: {all_categories}. CHOICE: {response} </s>
```

### 3.5 Hyperparameters (paper Table 5)

| Model | Epochs | Learning rate | Weight decay | Compute |
|---|---|---|---|---|
| FastFit | 10 | 5e-5 | 0.01 | 5–15 min to train; fast inference |
| mBERT | 5 | 5e-5 | 0.01 | 0.5 h to train; fast inference |
| Qwen2.5-7B | 3 (10 for TAXI1500) | 1e-5 | 0.001 | 6 h train + 3 h inference |

Model selection used **evaluation loss**, not test performance.

### 3.6 Evaluation
- **Accuracy** is the reported measure.
- Macro-F1 was computed too, but the authors report only accuracy because the two always
  agreed in ordering. We record both anyway (project rule 10); this is an addition, not a
  contradiction.
- Hardware: Apple M1 Pro / 32 GB for zero-shot, FastFit, logistic regression and synthetic
  generation; Colab T4 for mBERT on the two large datasets; V100 for Qwen SFT.

## 4. Results as reported by the paper

> Transcribed from the paper. **These are the authors' numbers, not ours.** Our own results
> live under `results/ours/` and must never be merged into these tables.

### 4.1 Single-source synthetic data (Table 3, average accuracy across languages)

| Task | GPT-4 | Aya-Expanse | Qwen2.5 |
|---|---|---|---|
| TWITTER | 0.50 | 0.47 | **0.51** |
| TAXI1500 | 0.43 | 0.40 | **0.43** |
| INTENT | **0.53** | 0.48 | 0.40 |
| SCENARIO | **0.65** | 0.64 | 0.59 |

### 4.2 Cross-language disparity (Table 4, absolute best-minus-worst gap)

| Method | Difference |
|---|---|
| Zero-shot Aya-Expanse-8B | 11% |
| Zero-shot GPT-4 | 13% |
| Zero-shot Aya-8B | 13% |
| Zero-shot Qwen2.5-7B | 15% |
| SFT-real | 17% |
| SFT-synthetic | 17% |
| Logistic regression-synthetic | 17% |
| FastFit | 22% |
| BERT-FT-real | 23% |
| Logistic regression-real | 29% |
| BERT-FT-synthetic | 12% |

### 4.3 Full per-language results

Tables 6–9 of the paper give every method for every language. Columns are, in order:
zero-shot (GPT-4, Aya-8B, Qwen2.5-7B, Aya-Expanse-8B), few-shot (FastFit), real data
(Logistic Reg., BERT-FT, SFT), synthetic data (Logistic Reg., BERT-FT, SFT).

**SENTIMENT / TWITTER (Table 6)**

| Language | GPT4 | Aya:8B | Qwen2.5:7B | Aya-Exp:8B | FastFit | LR-real | BERT-real | SFT-real | LR-syn | BERT-syn | SFT-syn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Arabic | 0.68 | 0.59 | 0.58 | 0.68 | 0.44 | 0.61 | 0.45 | 0.65 | 0.51 | 0.50 | 0.53 |
| English | 0.71 | 0.65 | 0.68 | 0.71 | 0.59 | 0.65 | 0.65 | 0.58 | 0.59 | 0.52 | 0.51 |
| French | 0.54 | 0.52 | 0.47 | 0.58 | 0.45 | 0.65 | 0.68 | 0.57 | 0.61 | 0.49 | 0.45 |
| German | 0.67 | 0.62 | 0.65 | 0.70 | 0.55 | 0.70 | 0.68 | 0.73 | 0.57 | 0.47 | 0.60 |
| Hindi | 0.62 | 0.53 | 0.56 | 0.61 | 0.34 | 0.41 | 0.33 | 0.36 | 0.47 | 0.38 | 0.46 |
| Italian | 0.72 | 0.58 | 0.60 | 0.71 | 0.57 | 0.59 | 0.59 | 0.64 | 0.61 | 0.49 | 0.59 |
| Portuguese | 0.69 | 0.60 | 0.61 | 0.69 | 0.62 | 0.70 | 0.54 | 0.68 | 0.58 | 0.49 | 0.59 |
| Spanish | 0.68 | 0.59 | 0.62 | 0.69 | 0.48 | 0.65 | 0.56 | 0.61 | 0.61 | 0.49 | 0.52 |

**TAXI1500 (Table 7)**

| Language | GPT4 | Aya:8B | Qwen2.5:7B | Aya-Exp:8B | FastFit | LR-real | BERT-real | SFT-real | LR-syn | BERT-syn | SFT-syn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Arabic | 0.58 | 0.22 | 0.23 | 0.44 | 0.30 | 0.36 | 0.50 | 0.69 | 0.32 | 0.48 | 0.61 |
| English | 0.53 | 0.35 | 0.40 | 0.43 | 0.66 | 0.74 | 0.77 | 0.69 | 0.54 | 0.51 | 0.33 |
| French | 0.54 | 0.23 | 0.34 | 0.41 | 0.69 | 0.57 | 0.73 | 0.67 | 0.55 | 0.50 | 0.62 |
| German | 0.44 | 0.32 | 0.33 | 0.41 | 0.64 | 0.54 | 0.72 | 0.72 | 0.44 | 0.43 | 0.57 |
| Hindi | 0.52 | 0.27 | 0.35 | 0.44 | 0.61 | 0.50 | 0.64 | 0.62 | 0.46 | 0.41 | 0.57 |
| Italian | 0.49 | 0.27 | 0.32 | 0.38 | 0.56 | 0.56 | 0.35 | 0.60 | 0.48 | 0.49 | 0.43 |
| Portuguese | 0.47 | 0.30 | 0.41 | 0.45 | 0.56 | 0.55 | 0.69 | 0.56 | 0.58 | 0.48 | 0.47 |
| Spanish | 0.56 | 0.23 | 0.40 | 0.52 | 0.59 | 0.47 | 0.72 | 0.56 | 0.52 | 0.48 | 0.37 |

**INTENT (Table 8)**

| Language | GPT4 | Aya:8B | Qwen2.5:7B | Aya-Exp:8B | FastFit | LR-real | BERT-real | SFT-real | LR-syn | BERT-syn | SFT-syn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Arabic | 0.60 | 0.31 | 0.46 | 0.49 | 0.57 | 0.74 | 0.79 | 0.81 | 0.44 | 0.30 | 0.51 |
| English | 0.72 | 0.44 | 0.60 | 0.60 | 0.74 | 0.87 | 0.88 | 0.89 | 0.58 | 0.43 | 0.62 |
| French | 0.67 | 0.40 | 0.54 | 0.56 | 0.60 | 0.83 | 0.86 | 0.87 | 0.58 | 0.46 | 0.56 |
| German | 0.66 | 0.37 | 0.54 | 0.56 | 0.69 | 0.80 | 0.84 | 0.86 | 0.51 | 0.43 | 0.58 |
| Hindi | 0.66 | 0.39 | 0.50 | 0.52 | 0.69 | 0.83 | 0.82 | 0.86 | 0.57 | 0.45 | 0.59 |
| Italian | 0.68 | 0.41 | 0.55 | 0.56 | 0.68 | 0.83 | 0.85 | 0.86 | 0.57 | 0.43 | 0.57 |
| Portuguese | 0.66 | 0.38 | 0.54 | 0.55 | 0.70 | 0.83 | 0.86 | 0.88 | 0.56 | 0.44 | 0.59 |
| Spanish | 0.66 | 0.37 | 0.54 | 0.54 | 0.70 | 0.84 | 0.86 | 0.85 | 0.54 | 0.44 | 0.61 |

**SCENARIO (Table 9)**

| Language | GPT4 | Aya:8B | Qwen2.5:7B | Aya-Exp:8B | FastFit | LR-real | BERT-real | SFT-real | LR-syn | BERT-syn | SFT-syn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Arabic | 0.59 | 0.43 | 0.52 | 0.45 | 0.672 | 0.58 | 0.86 | 0.86 | 0.58 | 0.54 | 0.696 |
| English | 0.67 | 0.50 | 0.66 | 0.52 | 0.77 | 0.90 | 0.90 | 0.94 | 0.74 | 0.54 | 0.77 |
| French | 0.64 | 0.47 | 0.59 | 0.49 | 0.73 | 0.81 | 0.90 | 0.90 | 0.70 | 0.59 | 0.72 |
| German | 0.64 | 0.46 | 0.60 | 0.50 | 0.71 | 0.79 | 0.90 | 0.91 | 0.68 | 0.59 | 0.82 |
| Hindi | 0.62 | 0.45 | 0.54 | 0.48 | 0.71 | 0.55 | 0.87 | 0.87 | 0.70 | 0.57 | 0.79 |
| Italian | 0.63 | 0.46 | 0.59 | 0.49 | 0.73 | 0.80 | 0.89 | 0.89 | 0.68 | 0.61 | 0.75 |
| Portuguese | 0.63 | 0.46 | 0.61 | 0.50 | 0.73 | 0.81 | 0.90 | 0.93 | 0.67 | 0.57 | 0.76 |
| Spanish | 0.63 | 0.46 | 0.60 | 0.48 | 0.72 | 0.79 | 0.89 | 0.90 | 0.68 | 0.56 | 0.74 |

## 5. The paper's own conclusions

1. **Sentiment is the exception.** Zero-shot LLMs (GPT-4 and Aya-Expanse-8B) match or beat
   classifiers trained on full human-labelled data. For the other three tasks, GPT-4 leads
   among zero-shot methods but trails full-data training, increasingly so as the label count
   grows.
2. **Few-shot fine-tuning beats zero-shot** on 3 of 4 tasks (up to ~10% gain; >5% over
   zero-shot GPT-4 on TAXI1500 and SCENARIO), but not on sentiment. On TAXI1500 it even
   edges past full-data training.
3. **Synthetic data beats zero-shot open LLMs** on all tasks except sentiment, matches GPT-4
   on INTENT, and beats GPT-4 on SCENARIO — at a one-time generation cost instead of
   per-call prompting cost. **Combining sources beats any single source.**
4. **Full labelled data still wins overall**, and SFT is best with many categories, but
   logistic regression remains a strong, cheap choice for simpler tasks.
5. **Large cross-language gaps everywhere** (11–29%), worst for logistic regression on real
   data. Non-Roman scripts (Arabic, Hindi) generate slower and cost more tokens — up to 3× —
   which matters for proprietary generation.

## 6. Limitations the authors acknowledge

- One zero-shot prompt and one generation prompt only; few-shot prompting not explored.
- Few-shot fine-tuning may be sample-sensitive; not systematically tested for variance.
- No multilingual (jointly trained) classifiers; no language-specific model choices.
- No qualitative analysis of outputs or of the generated synthetic data.
- No repeated runs with different initialisations; hyperparameters barely tuned.
- All datasets are **short texts** — results may not transfer to long documents.
- Possible training-data contamination for the zero-shot models is discussed but unresolved.
- 7 of 8 languages are Indo-European, so generalisation to other families is unproven.
