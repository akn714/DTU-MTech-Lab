"""
AI531C T08 - Learn Your First Kernel
Run: python T08_studio.py

Demonstrates:
1. 3x3 least-squares kernel learning for deblurring
2. 3x3 least-squares kernel learning for denoising
3. Patch-count stability
4. Optional 5x5 deblurring extension

This is a self-contained implementation of the T08 concepts.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def psnr(ref, img):
    ref = ref.astype(float)
    img = img.astype(float)
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


def make_image():
    """Create a deterministic grayscale test image."""
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]

    img = (
        90
        + 45 * np.sin(x / 17)
        + 35 * np.cos(y / 23)
    ).astype(float)

    # Add structures/edges so filtering has something to learn.
    img[45:205, 65:190] += 55
    img[85:165, 105:150] -= 70

    return np.clip(img, 0, 255).astype(np.uint8)


def patches(image, size=3, max_patches=20000):
    """Return flattened image patches and their centre-pixel targets."""
    r = size // 2
    p = np.pad(image.astype(np.float32), r, mode="reflect")

    X = []
    y = []

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            patch = p[row:row + size, col:col + size]
            X.append(patch.reshape(-1))
            y.append(image[row, col])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) > max_patches:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_patches, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def learn_kernel(source, target, size=3, max_patches=20000):
    """Learn kernel weights using least squares: Xw ~= y."""
    X, y = patches(source, size, max_patches)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w.reshape(size, size)


def apply(image, kernel):
    return cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel.astype(np.float32),
        borderType=cv2.BORDER_REFLECT_101
    )


# ---------------------------------------------------------
# Create clean reference image
# ---------------------------------------------------------
clean = make_image()


# =========================================================
# ACT 1: DEBLURRING
# =========================================================

blurred = cv2.GaussianBlur(clean, (9, 9), 2.0)

# Learn a 3x3 kernel: blurred patches -> clean pixels.
K_deblur = learn_kernel(blurred, clean, 3, 20000)
learned_deblur = np.clip(apply(blurred, K_deblur), 0, 255)

# Fixed sharpen baseline.
K_sharp = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

fixed_sharpen = np.clip(apply(blurred, K_sharp), 0, 255)

print("=" * 55)
print("ACT 1 - DEBLURRING")
print("=" * 55)
print(f"Input blurred:       {psnr(clean, blurred):.2f} dB")
print(f"Fixed sharpen:       {psnr(clean, fixed_sharpen):.2f} dB")
print(f"Learned 3x3:         {psnr(clean, learned_deblur):.2f} dB")

print("\nLearned deblur kernel:")
print(np.round(K_deblur, 3))


# =========================================================
# ACT 2: DENOISING
# =========================================================

rng = np.random.default_rng(1)
noise = rng.normal(0, 20, clean.shape)
noisy = np.clip(clean.astype(float) + noise, 0, 255).astype(np.uint8)

# Tuned Gaussian baseline.
gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.2)

# Learn noisy patches -> clean centre pixels.
K_denoise = learn_kernel(noisy, clean, 3, 20000)
learned_denoise = np.clip(apply(noisy, K_denoise), 0, 255)

print("\n" + "=" * 55)
print("ACT 2 - DENOISING")
print("=" * 55)
print(f"Noisy input:         {psnr(clean, noisy):.2f} dB")
print(f"Tuned Gaussian:      {psnr(clean, gaussian):.2f} dB")
print(f"Learned 3x3:         {psnr(clean, learned_denoise):.2f} dB")
print(f"Denoise weight sum:  {K_denoise.sum():.4f}")

print("\nLearned denoise kernel:")
print(np.round(K_denoise, 4))


# =========================================================
# PATCH COUNT STABILITY
# =========================================================

print("\n" + "=" * 55)
print("ACT 1 - PATCH COUNT STABILITY")
print("=" * 55)

for n in [2000, 5000, 10000, 20000]:
    K = learn_kernel(blurred, clean, 3, n)
    result = np.clip(apply(blurred, K), 0, 255)
    print(f"{n:5d} patches -> {psnr(clean, result):.2f} dB")


# =========================================================
# EXTENSION: 5x5 DEBLUR
# =========================================================

K5 = learn_kernel(blurred, clean, 5, 20000)
deblur5 = np.clip(apply(blurred, K5), 0, 255)

print("\n" + "=" * 55)
print("EXTENSION - LEARNED 5x5 DEBLUR")
print("=" * 55)
print(f"Learned 5x5:         {psnr(clean, deblur5):.2f} dB")


# =========================================================
# SAVE VISUAL RESULTS
# =========================================================

fig, ax = plt.subplots(1, 4, figsize=(13, 4))
items = [
    (clean, "Clean target"),
    (blurred, f"Blurred\n{psnr(clean, blurred):.2f} dB"),
    (fixed_sharpen, f"Fixed sharpen\n{psnr(clean, fixed_sharpen):.2f} dB"),
    (learned_deblur, f"Learned 3x3\n{psnr(clean, learned_deblur):.2f} dB"),
]

for a, (im, title) in zip(ax, items):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(title)
    a.axis("off")

plt.suptitle("T08 Act 1 - Deblurring")
plt.tight_layout()
plt.savefig("T08_deblur_results.png", dpi=150)
plt.close()


fig, ax = plt.subplots(1, 4, figsize=(13, 4))
items = [
    (clean, "Clean target"),
    (noisy, f"Noisy\n{psnr(clean, noisy):.2f} dB"),
    (gaussian, f"Tuned Gaussian\n{psnr(clean, gaussian):.2f} dB"),
    (learned_denoise, f"Learned 3x3\n{psnr(clean, learned_denoise):.2f} dB"),
]

for a, (im, title) in zip(ax, items):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(title)
    a.axis("off")

plt.suptitle("T08 Act 2 - Denoising")
plt.tight_layout()
plt.savefig("T08_denoise_results.png", dpi=150)
plt.close()

print("\nDone.")
print("Saved: T08_deblur_results.png")
print("Saved: T08_denoise_results.png")