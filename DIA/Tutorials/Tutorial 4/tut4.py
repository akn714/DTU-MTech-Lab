"""AI531C T04 studio — Noise Fingerprint Lab (Tue 25 Aug; COMPACT 25-min studio —
Assessment Test I occupies the first half of this slot per the lesson plan).
INPUT: camera; three corrupted versions generated with a FIXED seed.
OPERATION: for each mystery image: flat-patch histogram -> identify the noise ->
  estimate its parameter -> check against truth.
PARAMETERS: seed 11; Gaussian sigma 18; salt&pepper 4%; Poisson scale 12.
EXPECTED OUTPUT: printed estimates (asserted near truth) + expected_outputs/T04_panel.png.
INTERPRETATION: the flat patch is the stethoscope — diagnosis before treatment (L11 next week).
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/expected_outputs", exist_ok=True); os.makedirs(f"{HERE}/inputs", exist_ok=True)
rng = np.random.default_rng(11)
img = data.camera().astype(np.float64)
flat = (slice(20, 60), slice(150, 350))          # sky strip: flat-ish

A = np.clip(img + rng.normal(0, 18, img.shape), 0, 255)              # gaussian
B = img.copy(); m = rng.random(img.shape); B[m < 0.02] = 0; B[m > 0.98] = 255   # s&p 4%
C = rng.poisson(img / 12.0) * 12.0                                    # poisson
for name, x in zip("ABC", (A, B, C)): cv2.imwrite(f"{HERE}/inputs/mystery_{name}.png", np.clip(x,0,255).astype(np.uint8))

sigA = A[flat].std(); fracB = ((B[flat] == 0) | (B[flat] == 255)).mean()
darkC = C[img < 60].std(); brightC = C[img > 180].std()
print(f"[1] mystery A flat-patch std = {sigA:.1f}   (truth: Gaussian sigma 18)")
print(f"[2] mystery B extreme-pixel fraction = {fracB:.3f} (truth: s&p 4% total)")
print(f"[3] mystery C: std in DARK regions {darkC:.1f} vs BRIGHT {brightC:.1f} — signal-dependent!")
assert abs(sigA - 18) < 3 and 0.02 < fracB < 0.06 and brightC > darkC

fig, ax = plt.subplots(2, 3, figsize=(12.5, 6))
for j, (x, t) in enumerate(zip((A, B, C), ("mystery A", "mystery B", "mystery C"))):
    ax[0, j].imshow(x, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(t, fontsize=13); ax[0, j].axis("off")
    ax[1, j].hist(x[flat].ravel(), bins=60, color="#17365D"); ax[1, j].set_title("flat-patch histogram", fontsize=12)
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T04_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T04_panel.png")

def support():
    print("[SUPPORT] identify only mystery B: print the fraction of pure-black and pure-white")
    print("  pixels in the flat strip. Which classical noise makes pixels JUMP to the extremes?")

def extension():
    med = cv2.medianBlur(np.clip(B,0,255).astype(np.uint8), 3).astype(np.float64)
    print(f"[EXTENSION] 3x3 median on mystery B: flat-strip extreme fraction {((med[flat]==0)|(med[flat]==255)).mean():.4f}")
    print("  One week EARLY: why did the median annihilate this noise? (L11 explains.)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()