# SparConvLow — Project Implementation Plan

## 1. Paper and project objective

**Paper:** *Integrating Convolution and Sparse Coding for Learning Low-Dimensional Discriminative Image Representations*.

- **Authors:** Xian Wei, Yingjie Liu, Xuan Tang, Shui Yu, and Mingsong Chen.
- **Publication:** IEEE Transactions on Neural Networks and Learning Systems, volume 36, number 7, July 2025, pages 12483–12496. First published online on September 18, 2024.
- **DOI:** https://doi.org/10.1109/TNNLS.2024.3453374
- **Local source:** `../Integrating_Convolution_and_Sparse_Coding_for_Learning_Low_Dimensional.pdf`
- **Author-provided code:** https://github.com/MVPR-Group/SparConvLow

**Objective:** Build a small-scale, reproducible implementation of SparConvLow that learns compact, discriminative image embeddings by jointly training a CNN, an orthogonal feature projection, an elastic-net sparse dictionary, and a final discriminative projection. Compare the resulting representations against a conventional CNN under the same data and evaluation protocol.

This is an image-representation and classification project, not a license-plate detector or an OCR pipeline. The initial deliverable is a supervised CIFAR-10 experiment using the paper's LDA-based variant, **SparLDA**. A reduced experiment should establish correctness before attempting broader reproduction.

**Status:** Planning only. The files, commands, tests, and experiments below describe future implementation work; they have not been implemented or run.

## 2. Scope and expected deliverables

### Required implementation

1. A reproducible CIFAR-10 data pipeline and ResNet-18 baseline.
2. The complete CNN → projection → sparse coding → discriminative projection pipeline.
3. Differentiable elastic-net sparse coding, including a tested implicit backward pass.
4. A stabilized trace-ratio objective and manifold-constrained updates.
5. Evaluation using k-nearest neighbors and an SVM trained on learned embeddings.
6. Baseline comparisons, essential ablations, checkpoints, and experiment logs.
7. A command-line prediction demo and a final results report.

### Optional extensions, in priority order

1. Fashion-MNIST as a second dataset.
2. Semi-supervised SparSDA on CIFAR-10.
3. SVHN, ResNet-34, or the Laplacian-eigenmap-based SparLE variant.
4. Caltech transfer-learning experiments if compute and time permit.

**Out of initial scope:** Full ImageNet reproduction, all backbone comparisons, face-recognition benchmarks, unsupervised learning, and multimodal extensions. The paper's ImageNet experiments used four 2080 Ti GPUs and substantially larger training budgets; they are not a sensible starting point for this project.

## 3. What must be preserved from the paper

| Component | Paper-grounded requirement | Main source |
|---|---|---|
| CNN extractor | Produce a fixed-length convolutional feature vector for each image | Section IV; Figure 1 |
| First projection | Reduce feature dimension before sparse coding; maintain orthonormal columns in `U` | Equations (9), (11), and (12) |
| Sparse coding | Solve an elastic-net reconstruction problem over a learned dictionary | Equations (4) and (12) |
| Dictionary constraint | Maintain unit-norm dictionary atoms | Equation (5) |
| Final projection | Learn an orthogonal discriminative subspace of the sparse coefficients | Equations (13)–(19) |
| Training objective | Jointly maximize a stabilized discriminative trace quotient | Equations (19) and (21) |
| Optimization | Deliver gradients through sparse coding to the CNN and preserve manifold constraints | Section V; Algorithm 1 |
| Downstream classification | Apply k-NN or SVM to final low-dimensional embeddings | Figure 1; Section VI |

A frozen CNN followed by independently fitted PCA, sparse coding, and LDA is a useful baseline, but **is not the paper's end-to-end method**. Likewise, replacing the trace ratio with only cross-entropy or using a nondifferentiable sparse solver in the training path would not reproduce the core contribution.

## 4. Model architecture and consistent tensor notation

Use **sample rows** throughout the implementation. The paper switches matrix orientation in places; explicit shape contracts will prevent accidental transposes.

For a batch of `b` images:

| Symbol | Shape | Meaning |
|---|---|---|
| `I` | `b × channels × height × width` | Input images |
| `X = CNN_theta(I)` | `b × m` | Convolutional features |
| `U` | `m × l` | First projection; `U.T @ U = I_l` |
| `H = X @ U` | `b × l` | Dimension-reduced features |
| `D` | `l × k` | Dictionary; each column has unit norm |
| `Phi` | `b × k` | Elastic-net sparse coefficients |
| `V` | `k × d` | Final orthonormal basis; `V.T @ V = I_d` |
| `Y = Phi @ V` | `b × d` | Final discriminative embeddings |

The prediction path is:

`image → CNN → XU → elastic-net codes → Phi V → fitted k-NN/SVM → class`

For CIFAR-10, the LDA-based target is `d = c - 1 = 9`, following Section VI-A. Maintain `l < m`, `k > l`, and `d < k`.

### CNN feature choice

- Start with ResNet-18 adapted to 32 × 32 images: a 3 × 3, stride-1 stem and no initial max-pool. This is a **proposed small-image adaptation**, to be checked against the reference implementation.
- Prefer flattening the final convolutional feature map to preserve the paper's high-dimensional-feature motivation. With the stated CIFAR stem, a `512 × 4 × 4` map gives `m = 8192`; assert the actual shape at runtime.
- The paper's Table I lists `m = 25088` for ResNet-18/34, but this must not be assumed for every input resolution or feature hook.
- Offer an explicit low-compute mode using global-average-pooled features (`m = 512`). Label its results as a compute-reduced variant, not an identical reproduction.
- All raw-feature baselines must use the same chosen feature hook. Record the hook, input resolution, initialization, and dimension in each run.

### Avoid a large projector matrix

The paper represents the final Grassmann point as `P = V @ V.T`. Store and optimize the basis `V`, not a dense `k × k` projector. Compute trace quantities through `Y = Phi @ V`. This preserves the subspace representation while avoiding quadratic projector storage.

## 5. Sparse coding layer

For each projected feature `h`, solve Equation (12):

```text
phi*(h, D) = argmin_phi
    0.5 * ||h - D phi||_2^2
    + lambda1 * ||phi||_1
    + 0.5 * lambda2 * ||phi||_2^2
```

Require `lambda1 >= 0` and `lambda2 > 0`. The positive quadratic regularizer ensures a unique solution, although very small values can still produce numerically ill-conditioned backward systems.

### Forward solver

1. Implement batched ISTA as a transparent correctness reference.
2. Add FISTA for practical training, using a step size no larger than `1 / (||D||_2^2 + lambda2)`.
3. Expose maximum iterations, stopping tolerance, support threshold, and initialization through configuration.
4. Check the elastic-net KKT residual, not just the change between successive coefficients.
5. Initially run the numerical solve outside the autograd tape and supply a custom backward pass. A finite unrolled ISTA/FISTA network may be used for debugging, but must be labeled an approximate variant.
6. Avoid forming `D.T @ D` for very large dictionaries; use residual-based matrix products and small active-set systems instead.

### Implicit differentiation

For a locally fixed active set `Lambda`, define:

```text
M = D_Lambda.T @ D_Lambda + lambda2 * I
phi_Lambda = solve(M, D_Lambda.T @ h - lambda1 * sign(phi_Lambda))
```

Given upstream coefficient gradient `g`, solve `M beta_Lambda = g_Lambda` and set inactive entries of `beta` to zero. Then:

```text
grad_h = D @ beta
grad_D = (h - D @ phi) @ beta.T - (D @ beta) @ phi.T
```

The outer products above describe one sample; accumulate contributions across the batch. Autograd can propagate `grad_h` through `H = X @ U` to `U`, `X`, and the CNN.

- Use linear solves, not explicit matrix inverses.
- Check gradients in float64 on toy examples whose supports are stable under small perturbations.
- Handle empty active sets explicitly.
- Log active-set sizes, KKT residuals, and conditioning problems.
- If a larger `lambda2` or numerical damping is necessary, expose and report it as a deviation from the paper rather than changing it silently.

**Critical acceptance condition:** A loss computed on `Y` must generate finite, nonzero gradients for the CNN, `U`, `D`, and `V` on a nondegenerate test batch. Calling a scikit-learn/SPAMS solver alone does not provide this differentiable training path.

## 6. Supervised discriminative objective

Implement the LDA-based trace ratio first. For labeled sparse coefficients, form standard between-class and within-class scatter statistics.

An equivalent sample-space formulation uses a one-hot label matrix `J`, with counts for the classes present in the batch:

```text
C = J @ diag(1 / class_counts) @ J.T
A = C - ones(b, b) / b
B = I_b - C
```

With the sample-row convention:

```text
R = trace(V.T @ Phi.T @ A @ Phi @ V)
    / (trace(V.T @ Phi.T @ B @ Phi @ V) + sigma * k)

loss = -R
```

The `sigma * k` term follows the **literal printed denominator** `tr(B_sparse P + sigma I_k)` in Equations (19)/(21). Verify the supplement and author code before strict reproduction. If they use a different placement of regularization, record that difference; adding `sigma`, `sigma * d`, or a regularizer inside `P` is not algebraically identical.

For efficiency, compute the same numerator and denominator directly in embedding space:

- Between-class term: `sum_c n_c * ||mean(Y_c) - mean(Y)||_2^2`.
- Within-class term: `sum_i ||Y_i - mean(Y_class(i))||_2^2`.

Test that this matches the sample-space expression. Do not change scatter normalization without also accounting for its effect on the regularization scale.

Use label-aware batches containing at least two classes and multiple examples per class. Reject or resample degenerate batches with no between-class signal. Labels are used for the training objective and classifier fitting, **not to generate an unseen image's embedding**.

Cross-entropy is appropriate for CNN pretraining and the conventional baseline. Keep the principal joint-training objective as the negative trace ratio. Any auxiliary cross-entropy term is an explicitly separate ablation.

## 7. Manifold-constrained optimization

Use ordinary SGD for CNN parameters and a separate, tested manifold-SGD implementation for `U`, `D`, and `V`.

### Constraint-preserving updates

1. **Stiefel projection `U`:** For the Euclidean metric, project gradient `G` to `G - U sym(U.T @ G)`, then retract with reduced QR.
2. **Oblique dictionary `D`:** Project to `G - D diag(D.T @ G)`, then normalize each updated dictionary column.
3. **Grassmann basis `V`:** Use horizontal gradient `G - V @ (V.T @ G)`, then retract with reduced QR.
4. Correct QR signs consistently so the diagonal of `R` is positive.
5. Never apply ordinary weight decay to constrained parameters without a justified manifold formulation.
6. Initially use no manifold momentum. Adding it later requires correct tangent-space transport; CNN momentum is independent.

Equation (25) gives a canonical-style Stiefel gradient whose form differs from the Euclidean-metric projection above. Select and document one metric consistently. A strict-reference mode must validate its formulas against the author implementation rather than mixing gradient and retraction conventions without explanation.

Because the objective is maximized, optimize `loss = -R` with descent steps. Test the sign on a fixed toy batch: an appropriately small accepted step should increase `R`. Do not expect stochastic minibatch training to improve monotonically at every step.

Monitor `||U.T U - I||`, maximum dictionary-column norm error, `||V.T V - I||`, gradient norms, and the trace denominator throughout training.

## 8. Dataset protocol and experiment integrity

### Primary dataset: CIFAR-10

- Use the official 50,000 training and 10,000 test images.
- Initially reserve a stratified 5,000-image validation set from the training split: 45,000 train / 5,000 validation / 10,000 test.
- Save split indices and seeds. Compute normalization statistics using the training partition only.
- Proposed training augmentation: random crop with padding and horizontal flip. The paper documents normalization but not every augmentation detail; verify the reference recipe before exact comparison.
- Use deterministic transforms for embedding extraction, classifier fitting, validation, and test evaluation.
- Select dictionary initialization samples only from the training partition.
- Fit PCA initialization, optional scalers, SVMs, and k-NN reference sets using training data only.
- Tune hyperparameters and choose checkpoints using validation results. Do not repeatedly select models against the test set.
- For a full-training comparison, optionally retrain on all 50,000 training images after locking the recipe, and do the same for each baseline.

### Secondary dataset: Fashion-MNIST

Use its official split and document the grayscale handling: a one-channel CNN stem or a consistently applied three-channel conversion. Keep the architecture and preprocessing identical across methods being compared.

### Reproducibility

Record Python/package versions, hardware, configuration, split hash, seeds, CNN initialization, checkpoint provenance, solver tolerances, and code revision when available. Save RNG and optimizer states for resumable training. On Windows, use guarded process entry points and conservative data-loader worker counts.

## 9. Initial configurations and training stages

These are starting configurations, **not promises of reproducing the published accuracy**.

| Setting | Proposed project default | Paper information / caveat |
|---|---|---|
| Backbone | CIFAR-adapted ResNet-18 | Multiple CNN backbones are evaluated |
| Projected dimension `l` | 256 | Section VI-A uses 256 for tiny datasets |
| Dictionary atoms `k` | 1000; 512 in smoke mode | Main text mentions 1000; parameter sweep reaches 10,000 |
| Final embedding `d` | 9 for CIFAR-10 | `c - 1` for LDA/SDA |
| Dictionary initialization samples | 1000, class-balanced | Maintain `k <= n_trainD <= n_train` |
| `lambda1` | Start at 0.2; validate nearby values | CIFAR-10 discussion identifies 0.2 |
| `lambda2` | 1e-7 in paper-faithful mode | Section VI-A; monitor numerical conditioning |
| `sigma` | 1e-3 | Section VI-A |
| Batch size | 64, label-aware | Most paper experiments use 64 or 256 |
| CNN optimizer | SGD, initial LR 0.01, momentum 0.9 | Section VI-A; schedule must be verified |
| Manifold optimizer | Separate learning rates, initially no momentum | Proposed implementation detail |
| Training budget | Staged and validation-driven | Pretraining and joint budgets require reference verification |

### Stage A — Establish the CNN baseline

Train a conventional classifier to convergence under a documented schedule. Save the best validation checkpoint and its features. A proposed initial budget is 100 epochs, adjusted after profiling; this is not a paper-derived setting.

### Stage B — Initialize SparDR

- Initialize `U` with seeded random QR; optionally compare training-only PCA initialization.
- Initialize dictionary columns from nonzero projected training features and normalize them.
- Initialize `V` with seeded QR; optionally test a training-only discriminant-subspace initialization.
- Verify nontrivial code activity before optimization. Excessive sparsity or all-zero codes can stall the entire learning path.
- Save the initialization and freeze the CNN for a short SparDR warm-up, initially 5–10 epochs.

### Stage C — Joint optimization

Unfreeze the CNN and train all components through the tested implicit backward pass. Start with a proposed 20–50 joint-epoch budget, separate CNN/manifold learning rates, and validation-driven stopping. This schedule must remain distinguishable from the paper's references to approximately 50 **iterations**, whose pretraining context is not fully specified in the main article.

### Stage D — Fit downstream classifiers

Extract training embeddings from the selected checkpoint. Fit k-NN and a linear SVM initially; validate neighbor count and SVM regularization on validation embeddings. Verify the paper's actual SVM kernel and classifier settings before strict reproduction. Any embedding scaling must be training-fitted and recorded.

## 10. Baselines, ablations, and evaluation

### Required comparisons

| Experiment | Purpose |
|---|---|
| ResNet-18 + native classification head | Conventional CNN accuracy |
| Same CNN raw features + k-NN/SVM | Isolate classifier effects |
| CNN features + PCA to 9 dimensions + same classifier | Dimension-matched compact baseline |
| Frozen CNN + trained SparDR | Test the value of end-to-end learning |
| Full SparLDA-SparConvLow + k-NN/SVM | Main proposed implementation |

Give the baseline an additional-training control when the proposed model receives extra CNN fine-tuning. Match data splits, augmentation, initialization, classifier tuning effort, and evaluation transforms.

### Essential ablations

- `lambda1`: for example, `{0.1, 0.2, 0.4}`; compare accuracy and active-code fraction.
- Dictionary size: `{512, 1000}`, expanding to 2000 only if feasible.
- Projected dimension: `{128, 256}`.
- No sparse coding: orthogonal projections around the same backbone.
- Frozen versus jointly trained CNN.
- Constrained versus unconstrained updates on a small controlled run.

If resources are limited, prioritize the full method, CNN baseline, dimension-matched PCA baseline, frozen-CNN comparison, and sparsity sweep before adding new datasets.

### Metrics and outputs

Report top-1 accuracy, macro-F1, per-class performance, confusion matrix, final embedding dimension, active-code fraction, reconstruction residual, KKT residual, training time, embedding latency, and peak GPU memory. Define sparsity using a documented coefficient threshold.

For key comparisons, run at least three seeds and report mean and standard deviation. A 9-dimensional embedding compresses the **stored representation**, not necessarily the CNN parameters or total inference computation; sparse coding adds computational overhead that must be measured.

Planned visual analysis includes paired embedding visualizations and accuracy-versus-sparsity/dictionary-size plots. Treat t-SNE as qualitative evidence only; fit comparable visualization procedures and never claim that an attractive plot establishes classification performance.

### Published reference values, not project results

| Dataset / method | Reported accuracy | Source |
|---|---|---|
| CIFAR-10, ResNet-18 baseline | 94.95% ± 0.19 | Table II, PDF page 9 |
| CIFAR-10, ResNet-18 + SparLDA + SVM | 97.09% ± 0.14 | Table II, PDF page 9 |
| CIFAR-10, ResNet-18 + SparLDA + k-NN | 97.13% ± 0.13 | Table II, PDF page 9 |
| Fashion-MNIST, ResNet-18 baseline | 92.05% | Table V, PDF page 10 |
| Fashion-MNIST, ResNet-18 + SparLDA | 94.13% | Table V, PDF page 10 |

Do not treat these as guaranteed targets under a changed backbone, feature hook, split, or compute budget. If the implementation does not improve on the baseline, report the outcome and investigate it rather than selecting favorable test runs.

## 11. Correctness tests and release gates

Complete mathematical checks before long training runs.

| Test | Acceptance condition |
|---|---|
| Tensor contracts | All configured dimensions and matrix orientations are consistent |
| Elastic-net solver | Objective/KKT residual agrees with an independently validated reference on toy inputs |
| Sparse backward pass | Float64 finite-difference/gradcheck passes on stable-support examples; aim for relative error below 1e-4 |
| End-to-end connectivity | Finite, meaningful gradients reach CNN, `U`, `D`, and `V` |
| Scatter construction | Mean-based and sample-matrix trace ratios agree numerically |
| Objective sign | A small accepted fixed-batch step increases the ratio |
| Geometry | Orthogonality errors remain approximately below 1e-4 in float32; dictionary norm errors below 1e-5 |
| Subspace invariance | Replacing `V` by `VQ` for orthogonal `Q` leaves the trace objective unchanged |
| Numerical edge cases | Empty supports, repeated features, ill-conditioned active sets, and degenerate labels are handled explicitly |
| Inference independence | Embedding a new image does not require its label or an evaluation-set scatter matrix |
| Checkpoint round trip | Restored deterministic embeddings agree with saved-model embeddings |
| Data isolation | Initialization and classifier fitting never access validation/test labels as training data |

**Smoke test:** Use a stratified training subset, a small dictionary, and one or two short epochs. Verify finite losses, maintained constraints, complete backward flow, checkpoint saving, and classifier evaluation. Passing a smoke test establishes execution and basic integrity, not paper-level performance.

## 12. Technology stack and planned repository layout

### Stack

- Python 3.10 or 3.11 in a project-specific virtual environment.
- A mutually compatible, pinned PyTorch/torchvision pair, with CUDA support when available.
- NumPy, SciPy, scikit-learn, PyYAML, tqdm, pytest, and a logging/plotting library.
- A custom sparse solver and manifold optimizer as the main implementation; optional trusted numerical libraries for reference tests.
- Prefer local CSV/JSON and TensorBoard logging so experiments do not require an external service.

The public reference README lists an older stack including PyTorch 1.7.1, torchvision 0.8.2, and SPAMS. Its example command names differ from the scripts shown on the repository page. Inspect the actual source and dependencies before trying to run it. Do not assume that its original environment works unchanged on current Windows/Python installations, and verify licensing before copying implementation code.

### Planned layout — only `PLAN.md` exists at this stage

```text
SparConvLow/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline_cifar10.yaml
│   ├── smoke.yaml
│   ├── sparlda_cifar10.yaml
│   └── sparsda_cifar10.yaml
├── src/sparconvlow/
│   ├── data/                 # Datasets, transforms, saved splits, samplers
│   ├── models/               # Backbone, projections, sparse layer, full model
│   ├── losses/               # Scatter statistics and trace ratio
│   ├── optim/                # Manifold gradients and retractions
│   ├── evaluation/           # Embedding extraction, classifiers, metrics
│   ├── utils/                # Config, seeds, logging, checkpoints
│   ├── train_cnn.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/                    # Numerical, geometry, integration, leakage tests
├── data/                     # Downloaded datasets; excluded from Git
├── checkpoints/              # Model and classifier artifacts; excluded from Git
├── runs/                     # Per-run configs, logs, predictions, metrics
└── reports/                  # Final comparisons and reproduction notes
```

Future CLI interfaces should support commands such as:

```text
python -m sparconvlow.train_cnn --config configs/baseline_cifar10.yaml
python -m sparconvlow.train --config configs/smoke.yaml
python -m sparconvlow.train --config configs/sparlda_cifar10.yaml
python -m sparconvlow.evaluate --run-dir runs/<run-id> --split test
python -m sparconvlow.predict --run-dir runs/<run-id> --image <image-path>
pytest tests/
```

These are proposed interfaces. Install the future package in editable mode before using them; do not present these commands as runnable yet.

## 13. Resource plan and implementation milestones

A single CUDA GPU with roughly 8–12 GB VRAM and 16–32 GB system RAM is a reasonable starting target for the reduced experiment, subject to profiling. CPU-only execution should first target numerical tests and tiny smoke runs. No runtime estimate should be promised before benchmarking the sparse solver and QR updates.

Memory/time priorities:

- Do not allocate a dense `k × k` Grassmann projector or full-dataset `n × n` graph.
- Build supervised scatter statistics from batch means; use sparse/batched graphs for optional SDA.
- Cache features only while the CNN is frozen; invalidate caches after CNN updates.
- Extract evaluation features in batches. For example, 45,000 flattened 8192-dimensional float32 features occupy approximately 1.47 GB before classifier overhead.
- Profile FISTA iterations, active-set solves, and orthogonal retractions separately.
- Do not change input resolution, feature pooling, solver tolerance, or update frequency merely to improve runtime without recording the resulting methodological change.

### Suggested 6-week schedule

| Milestone | Tasks | Exit criterion |
|---|---|---|
| Week 1: Reference audit and baseline | Inspect author code/supplement, settle feature hook and conventions, build dataset splits, pretrain CNN | Documented reference decisions and reproducible baseline checkpoint |
| Week 2: Sparse layer | Implement ISTA/FISTA, KKT checks, implicit differentiation | Toy solver comparisons and gradient tests pass |
| Week 3: SparDR and geometry | Implement scatter objective, manifold updates, initialization, frozen-CNN training | Constraint tests and integrated smoke test pass |
| Week 4: Joint training | Connect CNN gradients, profile full pipeline, validate learning rates and stopping | Stable supervised end-to-end run with saved artifacts |
| Week 5: Controlled evaluation | Run key baselines, ablations, and repeated seeds | Fair comparison tables with runtime and sparsity measurements |
| Week 6: Packaging and optional extension | Finish CLI demo/report; attempt Fashion-MNIST or SDA only if core is complete | Reproducible project package and honest results report |

The schedule is an estimate, not a fixed deadline. The critical dependency is **sparse-layer numerical correctness → manifold correctness → end-to-end training → controlled evaluation**.

## 14. Semi-supervised extension: SparSDA

Attempt this only after supervised SparLDA passes its release gates.

1. Use a fixed labeled subset of the CIFAR-10 training partition, initially 4000 labels; withhold the remaining training labels from the learning procedure.
2. Build labeled scatter blocks and a symmetric neighborhood graph over labeled plus unlabeled training samples.
3. Construct `A` from the padded labeled between-class block and `B` from the padded within-class block plus `beta * graph_laplacian`, as in Section IV-C.
4. Verify the supplement's graph space, neighborhood count, weights, normalization, and sampling procedure. A batch-local graph over projected CNN features is a practical proposed variant, not an automatically exact reproduction.
5. Tune graph hyperparameters on validation data and compare against supervised training using the same labeled subset.
6. Keep the official test set out of the unlabeled training pool for this inductive project.

Do not implement SDA by training on pseudo-labels with an ordinary classification loss and calling that the paper's method; its graph-regularized trace-ratio construction must be explicit.

## 15. Reproduction risks and decisions to resolve

1. **Missing supplementary details:** The local 14-page article refers gradient derivations, initialization, additional experiment protocols, and other details to supplementary material. Obtain and inspect it before claiming a strict reproduction.
2. **Dictionary orthogonality wording:** The paper states unit-norm columns and also mentions pairwise orthogonality. An overcomplete `l × k` dictionary with `k > l` cannot have all columns mutually orthogonal. Implement the stated oblique/unit-column-norm constraint; do not impose impossible `D.T D = I_k`.
3. **Matrix-orientation and scatter notation:** Parts of the paper use inconsistent orientations/dimensions and class-scatter notation. Use verified shape contracts and standard scatter identities, documenting any correction against the reference implementation.
4. **Optimization conventions:** Resolve the maximization/descent sign, manifold metric, and exact denominator regularization before comparing results.
5. **Protocol ambiguity:** The CIFAR-10 parameter discussion mentions both 1000-atom and 10,000-atom settings, and the main text's iteration count does not fully specify pretraining. Preserve the distinction rather than inventing one universal training recipe.
6. **Numerical sensitivity:** Very small `lambda2`, support changes, poorly scaled features, and all-zero codes can invalidate naive differentiation or stall learning. Log diagnostics and validate changes explicitly.
7. **Reference-code portability:** Audit the actual scripts, solver conventions, dependency compatibility, and license; checking the README is not proof that the code is runnable or equivalent to every published experiment.
8. **Fairness and leakage:** Fit every learned preprocessing/classifier stage on training data and control for additional CNN training. Report all selected seeds, not only successful ones.

## 16. Definition of done

The project is complete when:

- [ ] The complete supervised pipeline is implemented with documented tensor shapes and equations.
- [ ] Sparse forward/backward, trace ratio, and manifold tests pass.
- [ ] The CNN and SparDR are demonstrably trained end to end.
- [ ] Data splits, configurations, checkpoints, and fitted classifiers are reproducible.
- [ ] Baseline, compact-baseline, frozen-CNN, and full-method comparisons are reported.
- [ ] Accuracy, representation size, sparsity, runtime, and memory tradeoffs are measured.
- [ ] A new image can be classified through the saved embedding model and classifier.
- [ ] The report separates published numbers, measured project results, and untested expectations.
- [ ] Every departure from the paper/reference protocol is recorded.

**Scientific success criterion:** Produce a correct, inspectable implementation and a fair empirical answer to whether sparse, orthogonally projected CNN representations improve compact image classification under the chosen resource budget. Matching or exceeding a published number is desirable, but is not a substitute for correctness and reproducibility.
