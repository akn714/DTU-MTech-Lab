"""AI531C T07 studio — The Measurement Studio (Tue 15 Sep, 2-3 PM).
INPUT: coins (auto-saved) — binarised at the T02 recipe.
OPERATION: clean -> per-object measurements (area, centroid, bounding box, aspect,
  extent) -> a size histogram -> failure round: sweep SE size until small coins die.
PARAMETERS: Otsu; SE sizes 5/25/35/45 (the gate closes near coin diameter).
EXPECTED OUTPUT: measurement table for 5 largest + survivor counts (asserted trend)
  + expected_outputs/T07_panel.png.
INTERPRETATION: morphology is a measurement PRE-FILTER; the SE size is a size gate.
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/expected_outputs", exist_ok=True); os.makedirs(f"{HERE}/inputs", exist_ok=True)
img = data.coins(); cv2.imwrite(f"{HERE}/inputs/coins.png", img)
_, bw = cv2.threshold(cv2.GaussianBlur(img,(5,5),1), 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

surv = []
for k in (5, 25, 35, 45):
    c = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k,k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
    keep = (stats[1:, cv2.CC_STAT_AREA] >= 200).sum()
    surv.append((k, keep)); print(f"[1] SE {k}x{k}: {keep} objects >= 200 px survive")
assert surv[0][1] >= surv[-1][1]

clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
n, lab, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:5] + 1
print("[2] five largest objects:  area  cx     cy     w   h  aspect extent")
for i in order:
    x,y,w,h,a = stats[i]
    print(f"    #{i:2d}  {a:5d}  {cents[i][0]:6.1f} {cents[i][1]:6.1f} {w:3d} {h:3d}  {w/h:5.2f}  {a/(w*h):5.2f}")
areas = stats[1:, cv2.CC_STAT_AREA]; areas = areas[areas>=200]
print(f"[3] coin-size stats: median area {np.median(areas):.0f} px, min {areas.min()}, max {areas.max()}")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
ax[0].imshow(lab, cmap="nipy_spectral"); ax[0].set_title("labelled objects", fontsize=13); ax[0].axis("off")
ax[1].hist(areas, bins=15, color="#17365D"); ax[1].set_title("area histogram (>=200 px)", fontsize=13)
ax[2].plot([s[0] for s in surv], [s[1] for s in surv], "o-", color="#A61C1C")
ax[2].set_xlabel("SE size"); ax[2].set_ylabel("survivors"); ax[2].set_title("the SE size gate", fontsize=13)
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T07_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T07_panel.png")

def support():
    print("[SUPPORT] run only step [2] with the provided clean mask; read the table and")
    print("  circle which column would best separate coins from a rectangular label (aspect? extent?).")

def extension():
    per = {i: cv2.arcLength(cv2.findContours((lab==i).astype(np.uint8), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_NONE)[0][0], True) for i in order}
    for i in order:
        a = stats[i, cv2.CC_STAT_AREA]; circ = 4*np.pi*a/per[i]**2
        print(f"[EXTENSION] object #{i}: circularity {circ:.3f} (1.0 = perfect disc)")
    print("  Which object is least circular, and does its image patch explain why?")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()
