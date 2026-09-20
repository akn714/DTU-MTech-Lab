"""AI531C T01 studio — First Contact (Tue 4 Aug, 2-3 PM).
INPUT: skimage built-ins (camera, coins, astronaut) — saved to inputs/ on first run.
OPERATION: load -> inspect (shape/dtype/range) -> slice a crop -> compute region stats
  -> darken/brighten by array arithmetic -> save. The array IS the image.
PARAMETERS: crop rows 100:200, cols 200:330; brightness offset +60 with clipping.
EXPECTED OUTPUT: printed block below + expected_outputs/T01_panel.png (asserted basics).
INTERPRETATION: every later operation in this course is arithmetic on these arrays.
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/inputs", exist_ok=True); os.makedirs(f"{HERE}/expected_outputs", exist_ok=True)

cam = data.camera(); coins = data.coins(); astro = data.astronaut()
cv2.imwrite(f"{HERE}/inputs/camera.png", cam); cv2.imwrite(f"{HERE}/inputs/coins.png", coins)
cv2.imwrite(f"{HERE}/inputs/astronaut.png", cv2.cvtColor(astro, cv2.COLOR_RGB2BGR))

print(f"[1] camera: shape {cam.shape} dtype {cam.dtype} range [{cam.min()}, {cam.max()}]")
print(f"[1] coins : shape {coins.shape} | astronaut: shape {astro.shape} (3 channels!)")
assert cam.shape == (512, 512) and cam.dtype == np.uint8

crop = cam[100:200, 200:330]
print(f"[2] crop rows 100:200, cols 200:330 -> shape {crop.shape}; mean {crop.mean():.1f}")
sky = cam[:60, :]; ground = cam[400:, :]
print(f"[3] sky mean {sky.mean():.1f} vs ground mean {ground.mean():.1f} — which is brighter, and does the image agree?")

bright = np.clip(cam.astype(np.int16) + 60, 0, 255).astype(np.uint8)
sat = (cam >= 196).sum()
print(f"[4] +60 brightness: {sat} pixels ({sat/cam.size:.1%}) hit the ceiling and CLIPPED — information gone.")

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, im, t in zip(ax, [cam, crop, bright, astro],
    ["camera (the array)", "a slice IS a crop", "+60, clipped", "3 channels (RGB)"]):
    a.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255); a.set_title(t, fontsize=13); a.axis("off")
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T01_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T01_panel.png")

def support():
    print("[SUPPORT] run these five lines one at a time:")
    print("  img = cv2.imread('inputs/camera.png', cv2.IMREAD_GRAYSCALE)")
    print("  print(img.shape); print(img.dtype); print(img.min(), img.max()); print(img[0, :5])")

def extension():
    r, g, b = astro[..., 0], astro[..., 1], astro[..., 2]
    print(f"[EXTENSION] channel means R {r.mean():.1f} G {g.mean():.1f} B {b.mean():.1f}")
    print("  Make a version with the red channel zeroed. What colour cast appears, and why?")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()
    