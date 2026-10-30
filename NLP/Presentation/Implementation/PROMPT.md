# Research Implementation Specification

I want to implement the research paper:

**"Text Classification in the LLM Era — Where do we stand?"**
Authors: Sowmya Vajjala and Shwetali Shimangaud.
Paper: arXiv:2502.11830.

## My hardware and constraints

- Laptop: Windows, 16 GB RAM, integrated GPU with 8 GB shared memory.
- Google Colab is available for GPU-intensive experiments.
- Use Python and beginner-friendly, modular code.
- Keep local experiments lightweight.
- Do not assume that I have a dedicated NVIDIA GPU.
- Do not start expensive API calls or long-running training jobs without explaining the cost and resource requirements first.

## Research objectives

Implement and compare:

1. Zero-shot classification using LLMs.
2. Few-shot fine-tuning using FastFit with 10 examples per label.
3. Synthetic-data generation using multiple LLM sources, followed by classifier training.
4. Supervised baselines:
   - Multilingual GTE embeddings + logistic regression.
   - Fine-tuned multilingual BERT.
   - Qwen2.5-7B instruction fine-tuning, if feasible with Colab resources.

The paper evaluates four tasks across eight languages:

- Twitter sentiment: 3 labels.
- Taxi1500: 6 labels.
- MASSIVE intent: 60 labels.
- MASSIVE scenario: 18 labels.

Languages: Arabic, English, French, German, Hindi, Italian, Portuguese and Spanish.

## Implementation rules

1. Read and summarize the paper's experimental methodology before writing code.
2. Check whether official supplementary code and datasets are available. Prefer the authors' implementation when accessible and compatible with my hardware.
3. Create a reproducible Python project with separate modules for data loading, model inference, training, evaluation and plotting.
4. Preserve the official train/validation/test splits. Never use test data for training, synthetic-data generation targets or hyperparameter selection.
5. Start with English Twitter sentiment classification and a logistic-regression baseline.
6. Validate the pipeline before adding more tasks and languages.
7. Use Google Colab for GPU training. Make training scripts configurable for batch size, epochs, model name and output directory.
8. Save intermediate predictions, synthetic datasets, checkpoints, configuration files and metrics to disk.
9. Make experiments resumable so interrupted runs do not require starting over.
10. Record accuracy, macro-F1, training time, inference time and hardware requirements.
11. Produce comparison tables and plots suitable for an academic implementation report.
12. Keep paper-reported results separate from our experimentally obtained results.
13. Document any deviations from the paper and do not fabricate results.
14. Add a README with setup instructions for Windows and Google Colab.

## Workflow

Work in small phases. First propose the architecture, directory structure, dependencies and implementation plan. Wait for my approval before generating the first module.

For each phase, explain what the code does, how to run it, what output to expect and how to verify correctness. Avoid unnecessary abstractions and do not generate the entire project in one response.