## Why do we use more than one layer in ANN?
- To learn hierarchical features and solve complex, non-linear problems that a single-layer network cannot handle.

## Demonstrate the loss function of logistic regression is convex.
For a **single training sample** `(x, y)`, where `y ∈ {0,1}`:

#### 1. Logistic Function

$$
\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}},\qquad z = \theta^T x
$$

The cost function is:

$$
J = -y\log(\hat{y})-(1-y)\log(1-\hat{y})
$$

This can be written as:

$$
J(z)=\log(1+e^z)-yz
$$

#### 2. First Derivative

$$
\frac{dJ}{dz}=\sigma(z)-y
$$

#### 3. Second Derivative

Since:

$$
\frac{d\sigma(z)}{dz}
=\sigma(z)(1-\sigma(z))
$$

we get:

$$
\boxed{\frac{d^2J}{dz^2}
=\sigma(z)(1-\sigma(z))}
$$

Why is $$ \sigma(z)(1-\sigma(z))\geq0\ $$

Because: $$ 0<\sigma(z)<1 $$

Therefore: $$ \sigma(z)>0 $$

and $$ 1-\sigma(z)>0 $$

Hence: $$ \boxed{\sigma(z)(1-\sigma(z))\geq0} $$

#### 4. Conclusion

A function is convex if its second derivative is non-negative:

$$
J''(z)\geq0
$$

Therefore:

$$
\boxed{\text{The logistic regression cost function is convex.}}
$$

**Key point:**

$$
\boxed{f''(x)\geq0 \Rightarrow f(x)\text{ is convex}}
$$
