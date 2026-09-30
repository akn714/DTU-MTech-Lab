### Autoencoder, Fully Connected Layers & Convolution

# 1. What is an Autoencoder?

An **autoencoder** is a neural network that learns to represent an input in a smaller form and then reconstruct the original input.

```text
Input Image
    ↓
  Encoder
    ↓
Compressed Representation
    ↓
  Decoder
    ↓
Reconstructed Image
```

Goal:

```text
Reconstructed Image ≈ Original Image
```

Common uses:
- Image reconstruction
- Image compression
- Denoising
- Feature extraction
- Dimensionality reduction
- Anomaly detection

---

# 2. Main Parts of an Autoencoder

```text
Encoder → Latent Representation → Decoder
```

## Encoder

The **encoder** converts the input into a smaller representation.

Example:

```text
784 → 128 → 64 → 32
```

The input has 784 values, while the encoder produces only 32 values.

```python
encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu")
])
```

## Latent Representation / Bottleneck

The **latent representation** is the compressed information produced by the encoder.

```text
784 → 128 → 64 → 32
                  ↑
              Bottleneck
```

The network learns these compact features automatically.

## Decoder

The decoder uses the latent representation to reconstruct the original input.

```text
32 → 64 → 128 → 784
```

```python
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid")
])
```

---

# 3. Complete Autoencoder

A simple autoencoder can be:

```text
784 → 128 → 64 → 32 → 64 → 128 → 784
                 ↑
             Bottleneck
```

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),

    # Encoder
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),

    # Decoder
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid")
])
```

---

# 4. How Does an Autoencoder Learn?

The important idea is:

**Input and target are the same.**

```python
model.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256
)
```

So:

```text
Original Image
      ↓
  Autoencoder
      ↓
Reconstructed Image
      ↓
Compare with original
      ↓
    Loss
      ↓
Update weights
```

---

# 5. Loss Function

For image reconstruction, **Mean Squared Error (MSE)** is commonly used.

```python
model.compile(
    optimizer="adam",
    loss="mse"
)
```

Conceptually:

```text
MSE = average((original pixel - reconstructed pixel)²)
```

A smaller MSE means the reconstructed image is closer to the original.

---

# 6. Fully Connected (Dense) Layer

A **fully connected layer**, or **Dense layer**, connects every neuron in one layer to every neuron in the next layer.

```text
Input neurons
 ● ───── ●
 ● ───── ●
 ● ───── ●
      ↓
Output neurons
```

Keras:

```python
tf.keras.layers.Dense(
    128,
    activation="relu"
)
```

This creates a layer with 128 neurons.

A neuron performs:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

then applies an activation:

```text
output = activation(z)
```

For ReLU:

```text
ReLU(x) = max(0, x)
```

---

# 7. Flattening an Image

A grayscale image can have shape:

```text
28 × 28
```

A Dense layer can receive it as a vector:

```text
28 × 28
   ↓
  784
```

because:

```text
28 × 28 = 784
```

Code:

```python
x = x.reshape(-1, 784)
```

After flattening:

```text
[pixel1, pixel2, pixel3, ..., pixel784]
```

**Limitation:** flattening removes the explicit 2D spatial arrangement of pixels.

---

# 8. Fully Connected Autoencoder

A fully connected autoencoder uses Dense layers for compression and reconstruction.

```text
Image
 ↓
Flatten
 ↓
784
 ↓
128
 ↓
64
 ↓
32       ← Bottleneck
 ↓
64
 ↓
128
 ↓
784
 ↓
Image
```

Example:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),

    # Encoder
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),

    # Decoder
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(784, activation="sigmoid")
])
```

**Main idea:**

> Flatten → Dense → Compress → Dense → Reconstruct

---

# 9. What is Convolution?

A **convolution** applies a small filter (kernel) over different regions of an image.

```text
Image
  ↓
Small filter moves across image
  ↓
Feature map
```

It can learn local visual patterns such as:

- Edges
- Curves
- Lines
- Corners
- Small shapes

---

# 10. Conv2D Layer

```python
tf.keras.layers.Conv2D(
    16,
    (3, 3),
    activation="relu",
    padding="same"
)
```

Meaning:

- `16` → number of filters
- `(3, 3)` → filter/kernel size
- `relu` → activation
- `same` → preserves spatial size when stride is 1

For a grayscale image:

```text
28 × 28 × 1
      ↓
    Conv2D
      ↓
28 × 28 × 16
```

---

# 11. Why Use Convolution for Images?

Instead of immediately doing:

```text
28 × 28 → 784
```

convolution first processes the image while preserving its spatial structure:

```text
28 × 28 × 1
      ↓
    Conv2D
      ↓
28 × 28 × 16
```

This lets the network learn local image features.

---

# 12. Max Pooling

**MaxPooling** reduces the spatial dimensions of feature maps.

```python
tf.keras.layers.MaxPooling2D((2, 2))
```

Example:

```text
28 × 28
   ↓
14 × 14
```

It takes the largest value from each small region.

Benefits:
- Reduces spatial size
- Reduces computation
- Keeps strong features
- Makes features less sensitive to small shifts

---

# 13. Flatten After Convolution

After convolution and pooling:

```text
14 × 14 × 16
      ↓
   Flatten
      ↓
3584
```

because:

```text
14 × 14 × 16 = 3584
```

Code:

```python
tf.keras.layers.Flatten()
```

---

# 14. Fully Connected + Convolution Autoencoder

A model can combine convolutional and fully connected layers.

### Encoder

```text
28×28×1
   ↓
Conv2D
   ↓
28×28×16
   ↓
MaxPooling
   ↓
14×14×16
   ↓
Flatten
   ↓
3584
   ↓
Dense 128
   ↓
Dense 32
   ↓
Bottleneck
```

### Decoder

```text
32
 ↓
Dense
 ↓
3584
 ↓
Reshape
 ↓
14×14×16
 ↓
UpSampling
 ↓
28×28×16
 ↓
Conv2D
 ↓
28×28×1
```

---

# 15. Example Code: Conv + Fully Connected Autoencoder

```python
model = tf.keras.Sequential([

    # Encoder
    tf.keras.layers.Conv2D(
        16, (3, 3),
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),  # Bottleneck

    # Decoder
    tf.keras.layers.Dense(
        14 * 14 * 16,
        activation="relu"
    ),

    tf.keras.layers.Reshape((14, 14, 16)),

    tf.keras.layers.UpSampling2D((2, 2)),

    tf.keras.layers.Conv2D(
        1, (3, 3),
        activation="sigmoid",
        padding="same"
    )
])
```

Compile and train:

```python
model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256
)
```

---

# 16. Reshape

`Reshape` changes the tensor shape without changing the number of values.

```python
tf.keras.layers.Reshape((14, 14, 16))
```

Since:

```text
14 × 14 × 16 = 3584
```

3584 values can be reshaped into:

```text
14 × 14 × 16
```

It is useful in the decoder to convert a vector back into a feature-map structure.

---

# 17. UpSampling

UpSampling increases spatial dimensions.

```python
tf.keras.layers.UpSampling2D((2, 2))
```

Example:

```text
14 × 14 × 16
      ↓
28 × 28 × 16
```

It helps the decoder return to the original image size.

---

# 18. Final Conv2D in the Decoder

The final convolution converts feature maps into the desired output channels.

```python
tf.keras.layers.Conv2D(
    1,
    (3, 3),
    activation="sigmoid",
    padding="same"
)
```

For a grayscale image:

```text
28 × 28 × 16
      ↓
28 × 28 × 1
```

When pixels are normalized to `[0, 1]`, sigmoid is useful because its output is approximately between 0 and 1.

---

# 19. Fully Connected vs Convolution

| Feature | Fully Connected | Convolution |
|---|---|---|
| Main layer | Dense | Conv2D |
| Image handling | Usually flatten first | Keeps spatial structure |
| Spatial information | Not explicitly preserved after flattening | Preserved |
| Feature learning | General combinations | Local visual patterns |
| Common use | Simple/general data | Image processing |

---

# 20. Three Architectures to Remember

### Basic Autoencoder

```text
Input
 ↓
Encoder
 ↓
Latent/Bottleneck
 ↓
Decoder
 ↓
Output
```

### Fully Connected Autoencoder

```text
28×28
 ↓
Flatten
 ↓
784
 ↓
Dense
 ↓
32
 ↓
Dense
 ↓
784
 ↓
28×28
```

### Convolution + Fully Connected Autoencoder

```text
28×28×1
 ↓
Conv2D
 ↓
MaxPooling
 ↓
Flatten
 ↓
Dense
 ↓
32
 ↓
Dense
 ↓
Reshape
 ↓
UpSampling
 ↓
Conv2D
 ↓
28×28×1
```

---

# 21. Quick Revision

| Term | Simple meaning |
|---|---|
| **Autoencoder** | Compresses and reconstructs data |
| **Encoder** | Compresses the input |
| **Latent representation** | Compressed learned features |
| **Bottleneck** | Smallest/most compressed representation |
| **Decoder** | Reconstructs the input |
| **Dense layer** | Every neuron connects to every neuron in the next layer |
| **Flatten** | Converts a multi-dimensional tensor into a vector |
| **Convolution** | Learns local image features using filters |
| **MaxPooling** | Reduces spatial size while keeping strong features |
| **Reshape** | Changes tensor shape without changing the number of values |
| **UpSampling** | Increases spatial dimensions |
| **MSE** | Measures reconstruction error |
| **Adam** | Optimizer used to update network weights |

---

# 22. Final Mind Map

```text
                    AUTOENCODER
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
          ENCODER                 DECODER
             │                       │
        Compress                  Reconstruct
             │                       │
             └────→ BOTTLENECK ←─────┘


FULLY CONNECTED
Image → Flatten → Dense → Bottleneck → Dense → Image


CONVOLUTION + FULLY CONNECTED
Image → Conv → Pool → Flatten → Dense
                                  ↓
                              Bottleneck
                                  ↓
                    Dense → Reshape → Upsample → Conv → Image
```

## Most important lines to remember

```text
Autoencoder:
Input → Compress → Bottleneck → Reconstruct

Fully Connected:
Image → Flatten → Dense → Bottleneck → Dense → Image

Conv + Fully Connected:
Image → Conv → Pool → Flatten → Dense → Bottleneck
      → Dense → Reshape → Upsample → Conv → Image
```
