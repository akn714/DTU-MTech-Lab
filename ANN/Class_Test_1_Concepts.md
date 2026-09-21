# ANN — Class Test 1 Concepts

Topics the paper covers: Linear Regression, Logistic Regression, Activation Functions,
Loss Functions, Perceptron, NumPy, Pandas.

---

## 1. Linear Regression
Model: `ŷ = w₁x₁ + w₂x₂ + b`, or in general `ŷ = wᵀx + b`.

- Called **linear** because it is linear *in its parameters* (`w`, `b`) — not because it uses one input.
- A model can take many features and still be linear.
- Adding a bias `b` only *shifts* the prediction; it never makes the model nonlinear.
- If all weights are 0, the prediction is just `b`, no matter the input.

## 2. Linearity in Parameters vs Linearity in Inputs
Two different things, and exams love to mix them up:

- **Linear in parameters** — no `w²`, no `w₁w₂`, no `log(w)`. Almost every model we train is linear in parameters (that's what makes gradient descent work).
- **Linear in inputs** — no `x²`, no `x₁x₂`.
- A model can be *nonlinear in inputs* but *linear in parameters*: `ŷ = w₀ + w₁x + w₂x²` is a parabola in `x`, but still a straight-line fit in `(w₀, w₁, w₂)`.
- That is exactly what **polynomial regression** is — nonlinear in inputs, linear in coefficients.

## 3. The Bias Term
- Adds a constant to the weighted sum: `z = wᵀx + b`.
- Increasing `b` by 1 increases `z` by 1, **regardless of the input**.
- Moves the decision boundary **without changing its orientation** (it shifts the line, doesn't rotate it).
- Bias terms alone can never introduce nonlinearity — a stack of biases is still just an offset.

## 4. Weights — What They Actually Do
For `z = w₁x₁ + w₂x₂ + b`:

- Increasing `w₁` does **not** always increase `z`. It increases `z` only when `x₁ > 0`; if `x₁ < 0` it *decreases* `z`.
- If `x₁ = 0`, changing `w₁` has **no effect at all** — the weight is multiplied by zero.
- The bias controls the *offset* of the boundary; the weights control its *slope/orientation*.

## 5. Pre-activation and Activation
For each layer:

- `z = Wx + b` → the **pre-activation** (raw weighted sum).
- `a = f(z)` → the **activation** (what the layer passes on).
- The next layer's weight matrix operates on `a`, **not** on the original input `x`.

## 6. Activation Functions — Overview
A nonlinear function applied after the linear combination. Without one, a network is just a linear model no matter how deep it is.

**Sigmoid** `σ(z) = 1/(1+e⁻ᶻ)`
- Output always in `(0, 1)` — reads as a probability.
- Nonlinear, so it bends the *output*, but it does **not** make the decision boundary of a single logistic unit nonlinear (see §12).
- Derivative is `σ(z)(1 − σ(z))`, which peaks at `z = 0` (value 0.25) and → 0 as `|z|` grows.
- **Saturation:** at `z = 10`, output ≈ 1 and the derivative is tiny. A tiny gradient means tiny parameter updates → the **vanishing gradient problem** in deep networks.
- Note: a small gradient can occur even when the prediction is highly confident — that's saturation, not success.
- Pushing `|z|` larger does **not** proportionally increase the output; the output flattens out.

**Step function**
- `1` if `z ≥ 0`, else `0`. Binary and non-differentiable.
- Used by the perceptron as a threshold operation on the weighted sum.
- Because it has no usable gradient, you can't train it with gradient descent — swapping sigmoid for a step changes the optimization problem completely.

**ReLU** `ReLU(z) = max(0, z)`
- Output is 0 for all negative inputs; output is `z` for positive inputs.
- Derivative is 1 for `z > 0` and 0 for `z < 0` — a constant slope, so it does not saturate on the positive side.
- **Nonlinear** — despite being a straight line for positive inputs, the bend at 0 is what makes it nonlinear overall.
- Output is **unbounded** above (not squeezed into 0–1).
- **Dying ReLU:** if a neuron's weights push it permanently negative, its gradient is always 0 and it stops learning entirely.

## 7. Why Nonlinearity Is Essential
- Stacking purely linear layers collapses: `h = W₁x + b₁`, `y = W₂h + b₂` becomes `y = (W₂W₁)x + (W₂b₁ + b₂)` — a **single affine transformation**.
- The composition is **not** nonlinear just because there are two weight matrices.
- Adding more linear layers (more depth) does **not** increase expressive power — it still equals one linear map.
- Biases don't help either.
- Therefore: a network of only linear transformations **cannot** approximate a nonlinear function by getting deeper. You need a nonlinear activation between layers.

## 8. Perceptron
`ŷ = 1` if `wᵀx + b ≥ 0`, else `0`.

- Essentially a **threshold on a weighted sum** of the inputs (step activation).
- Its decision boundary is a **hyperplane** — a line in 2D, a plane in 3D.
- It can learn any **linearly separable** problem.
- It does **not** require a sigmoid; a step function is enough.
- Changing only the bias shifts the boundary's **location**, not its orientation.

## 9. Linear Separability and XOR
- A problem is **linearly separable** if a single straight boundary can split the classes.
- **XOR is not linearly separable** — positives and negatives sit diagonally opposite each other.
- A single perceptron draws one linear boundary, so it **cannot** solve XOR.
- More training iterations don't help; a higher learning rate doesn't help; more data doesn't make XOR separable. The limitation is structural, not about tuning.
- A **multi-layer** network with nonlinear activations *can* represent XOR.

## 10. Multi-layer Networks and Representation Learning
- Hidden layers let the network build **intermediate representations** of the input before classifying.
- With a nonlinear activation (e.g. ReLU) between layers, the network can produce **nonlinear decision boundaries**.
- The first layer's `W₁` decides how the raw features are linearly combined before the nonlinearity.
- Stacking nonlinear layers is what buys expressive power — not depth alone.

## 11. Logistic Regression
`ŷ = σ(wᵀx + b)`.

- Output is constrained to `(0, 1)` and is interpreted as a **probability** for binary classification.
- The sigmoid is nonlinear, **but the decision boundary `wᵀx + b = 0` is still linear** in the original input space. The nonlinearity reshapes the *output*, not the boundary.
- Two logistic models with the same weights but different biases have **different** boundaries (shifted).
- It's trained with **gradient-based methods** (cross-entropy loss is differentiable).
- It can output a probability like 0.8 and, after thresholding at 0.5, still predict class 1 — the probability and the predicted class are different things.
- It **cannot** perfectly classify every binary dataset — only those that are linearly separable.
- Swapping sigmoid for a step function does **not** preserve the optimization problem — the step has no gradient.

## 12. Loss Functions — Overview
A number measuring how wrong the predictions are. Lower is better *on the training set* — but see §19.

**Mean Squared Error (MSE)** — `MSE = (1/n) Σ (yᵢ − ŷᵢ)²`
- Standard choice for **regression**.
- Always **non-negative**.
- Zero only when every prediction exactly matches its target.
- Squares the error, so it penalizes large errors *disproportionately*: an error of 4 costs 16 while an error of 2 costs 4 — four times as much, not twice.
- Not the right loss for binary classification.

**Binary Cross-Entropy (BCE)** — `L = −[y log(ŷ) + (1 − y) log(1 − ŷ)]`
- Standard choice for binary classification.
- For `y = 1`, the loss reduces to `−log(ŷ)`.
- Predicting a probability near 0 for a true class of 1 gives a **huge** loss.
- Confident **wrong** predictions are penalized very strongly; predicting 0.99 for a true class of 0 costs far more than predicting 0.60.
- Predicting 0.5 does **not** give zero loss (`−log 0.5 = 0.693`).
- Minimized when the predicted probability matches the true class.

## 13. Gradient Descent and Backpropagation
- Backpropagation computes the gradients; gradient descent uses them to update the parameters.
- If the gradient of the loss w.r.t. a weight is **zero**, that weight does **not change** in a standard update.
- **Backprop is not needed for the forward pass** — the forward pass just computes predictions from the current parameters.
- **Learning rate:** increasing it does *not* always reduce the number of iterations to reach the minimum. Too large and training oscillates or diverges.

## 14. Vanishing Gradients
- With saturating activations (sigmoid), gradients shrink as `|z|` grows.
- In deep networks, many small gradients multiplied together make early layers learn extremely slowly.
- This is a core motivation for ReLU.

## 15. Parameter Counting
Count weights and biases for every connection.

Worked example — 4 inputs → 5 hidden → 2 outputs, one bias per neuron:
- Input → hidden: `4 × 5 = 20` weights `+ 5` biases `= 25`
- Hidden → output: `5 × 2 = 10` weights `+ 2` biases `= 12`
- **Total = 37**

## 16. Layer Shapes (Matrix Dimensions)
For a batch of `N` samples with `F` features, and layers `Dense(d₁) → Dense(d₂) → …`:

- Input matrix shape: `(N, F)`
- Weight matrix between a layer of size `a` and the next layer of size `b`: `(a, b)`
- Output of each Dense layer: `(N, that layer's size)`

Example — batch 32, 6 features, `Dense(10) → Dense(5) → Dense(1)`:
- Input `(32, 6)` · first weight matrix `(6, 10)` · first hidden output `(32, 10)` · second weight matrix `(10, 5)` · final output `(32, 1)`
- Parameters: `(6×10 + 10) + (10×5 + 5) + (5×1 + 1) = 70 + 55 + 6 = 131`

## 17. NumPy — Shapes and Vectorization
- `X.shape = (100, 4)` means 100 samples, 4 features. `w.shape = (4,)` is a vector of 4 weights.
- `z = X @ w + b` produces shape `(100,)` — one value per sample.
- `@` is **matrix multiplication**, not element-wise multiplication (that's `*`).
- Matrix multiplication replaces an explicit Python loop over samples — far faster and the standard way to write it.

## 18. Pandas — Data Preparation
Before training, appropriate steps are:

- Separate **input features** from the **target variable**.
- Check for **missing values**.
- Inspect **data types** and unusual/outlier values.
- Check the **distribution** of numerical features.

Not appropriate: randomly changing feature values to "balance" the data — that destroys the signal.

## 19. Data Leakage, Train/Test Split, and Generalization
- The test set must stay unseen during preprocessing.
- If you **fit a scaler on the entire dataset before splitting**, information from the test set leaks into the scaling parameters. This is **data leakage**.
- Consequence: reported test performance becomes **overly optimistic**.
- Correct approach: fit the scaler on the **training data only**, then apply that same scaler to transform the test data. (Note: you *do* transform the test set with the training scaler — you just don't *fit* on it.)
- **Zero training error does not imply zero test error.** A model can memorize the training data and still fail on unseen data (overfitting).
- **A lower loss value does not always mean better generalization.**

## 20. Comparing Model Families
- **Step + linear** (perceptron) and **sigmoid + linear** (logistic regression) both have a **linear decision boundary** when thresholded at 0.5.
- They do **not** train the same way, because the step function is non-differentiable — no gradient to descend.
- Only the sigmoid version gives probability-like outputs in `(0, 1)`.
- **ReLU + sigmoid** (a small MLP) can represent **nonlinear** decision boundaries and learns internal representations before its final classification.
- Removing the ReLU from such a network while keeping the sigmoid leaves a purely linear model in the input — effectively logistic regression, not a perceptron with a step function.
