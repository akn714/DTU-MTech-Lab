"""AI531C T05 studio — The Filter Tournament (Tue 1 Sep, 2-3 PM).
INPUT: camera + two corruptions (Gaussian sigma 20; salt&pepper 4%), fixed seed.
OPERATION: run mean/Gaussian/median (5x5) on BOTH noises; fill the 3x2 PSNR table;
  then the parameter round: window 3/5/9 for the winner on each noise.
PARAMETERS: seed 5; window 5 default.
EXPECTED OUTPUT: PSNR table printed (bracket asserted) + expected_outputs/T05_panel.png.
INTERPRETATION: filters are hypotheses about the noise; match the hypothesis, win the cell.
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/expected_outputs", exist_ok=True)
rng = np.random.default_rng(5)
img = data.camera().astype(np.float64)
def psnr(x): return 10*np.log10(255**2/np.mean((img-x)**2))
gau = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
sp = img.copy().astype(np.uint8); m = rng.random(img.shape); sp[m<0.02]=0; sp[m>0.98]=255

F = { "mean":  lambda x, k: cv2.blur(x, (k, k)),
      "gauss": lambda x, k: cv2.GaussianBlur(x, (k, k), 0),
      "median":lambda x, k: cv2.medianBlur(x, k) }
print(f"noisy inputs: gaussian {psnr(gau):.1f} dB | salt&pepper {psnr(sp):.1f} dB")
table = {}
for fn, f in F.items():
    a, b = psnr(f(gau, 5).astype(np.float64)), psnr(f(sp, 5).astype(np.float64))
    table[fn] = (a, b); print(f"  {fn:6s} 5x5: on-gaussian {a:.1f} dB | on-s&p {b:.1f} dB")
assert table["median"][1] > table["mean"][1] and table["median"][1] > table["gauss"][1]
best_g = max(table, key=lambda k: table[k][0])
print(f"[bracket] gaussian-noise winner: {best_g}; s&p winner: median (assert holds)")
for k in (3, 5, 9):
    print(f"  parameter round, median {k}x{k} on s&p: {psnr(cv2.medianBlur(sp,k).astype(np.float64)):.1f} dB")

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, im, t in zip(ax, [sp, F["mean"](sp,5), F["median"](sp,5), F["median"](sp,9)],
    [f"s&p input {psnr(sp):.1f} dB", f"mean 5x5 {table['mean'][1]:.1f} dB",
     f"median 5x5 {table['median'][1]:.1f} dB", f"median 9x9 {psnr(cv2.medianBlur(sp,9).astype(np.float64)):.1f} dB"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t, fontsize=13); a.axis("off")
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T05_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T05_panel.png")

def support():
    print("[SUPPORT] run only the s&p column: mean vs median at 5x5. Record both PSNRs and")
    print("  write one line: what does the mean do to a black impulse that the median refuses to do?")

def extension():
    bil = cv2.bilateralFilter(gau.astype(np.uint8), 9, 60, 10)
    print(f"[EXTENSION] bilateral on gaussian noise: {psnr(bil.astype(np.float64)):.1f} dB — beat the 5x5 gaussian blur?")
    print("  Tune (d, sigmaColor, sigmaSpace) for 5 minutes; report your best and the setting.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()