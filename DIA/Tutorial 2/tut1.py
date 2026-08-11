"""AI531C — Tutorial T02 studio: Connectivity & Distance (Tue 11 Aug 2026, 2-3 PM).
INPUT: skimage coins() (saved to inputs/coins.png on first run) — no downloads needed.
OPERATION: binarise -> clean -> count components under 4- vs 8-adjacency ->
  measure areas -> compare the three distance rulers on the same image ->
  break the pipeline deliberately (threshold sweep) and watch objects merge.
PARAMETERS: Otsu threshold (auto); opening SE 5x5; connectivity in {4, 8};
  distance rulers DIST_L1 (D4), DIST_L2 (Euclidean), DIST_C (D8).
EXPECTED OUTPUT: expected_outputs/T02_panel.png + the printed block below
  (counts asserted -- if your numbers differ, your pipeline differs).
INTERPRETATION: the adjacency rule and the distance ruler are MODELLING CHOICES;
  today they visibly change how many objects exist and how far things are.
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/inputs", exist_ok=True); os.makedirs(f"{HERE}/expected_outputs", exist_ok=True)

img = data.coins()
cv2.imwrite(f"{HERE}/inputs/coins.png", img)

# --- Step 1 (10-25 min): binarise and clean --------------------------------
t, bw = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 1), 0, 1,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
print(f"[1] Otsu threshold t = {t:.0f}")

# --- Step 2 (25-35 min): count under both adjacency rules ------------------
n4, _ = cv2.connectedComponents(clean, connectivity=4)
n8, lab = cv2.connectedComponents(clean, connectivity=8)
n8s, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]
big = (areas >= 200).sum()
print(f"[2] components: 4-adjacency {n4-1} | 8-adjacency {n8-1} | >=200 px {big}")
print("    QUESTION: the image has 24 coins. Is every component a coin? Find the impostor.")

# staircase from PCM-2, resolved live:
g = np.eye(4, dtype=np.uint8)
s4, _ = cv2.connectedComponents(g, connectivity=4)
s8, _ = cv2.connectedComponents(g, connectivity=8)
print(f"[2b] PCM-2 staircase: 4-adj {s4-1} components, 8-adj {s8-1}")
assert (s4-1, s8-1) == (4, 1)

# --- Step 3 (35-40 min): three distance rulers on the SAME shapes ----------
d4 = cv2.distanceTransform(clean, cv2.DIST_L1, 3)   # D4  (city-block)
de = cv2.distanceTransform(clean, cv2.DIST_L2, 3)   # DE  (Euclidean)
d8 = cv2.distanceTransform(clean, cv2.DIST_C, 3)    # D8  (chessboard)
print(f"[3] max in-coin distance: D4 {d4.max():.0f} | DE {de.max():.1f} | D8 {d8.max():.0f}")
assert d4.max() >= de.max() >= d8.max()             # the ruler ordering, live

# --- Step 4 (40-50 min): break it — threshold sweep merges coins -----------
merge_report = []
for dt in (-40, -20, 0, +20):
    _, b = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 1), t + dt, 1, cv2.THRESH_BINARY)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _ = cv2.connectedComponents(b, connectivity=8)
    merge_report.append((t + dt, n - 1))
print("[4] threshold -> component count:", merge_report)

# --- Panel -----------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.8))
ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"input (Otsu t={t:.0f})", fontsize=13)
ax[1].imshow(lab, cmap="nipy_spectral"); ax[1].set_title(f"8-adj components: {n8-1} — all coins? find the impostor", fontsize=13)
ax[2].imshow(de, cmap="magma"); ax[2].set_title(f"Euclidean distance (max {de.max():.1f})", fontsize=13)
xs = [m[0] for m in merge_report]; ys = [m[1] for m in merge_report]
ax[3].plot(xs, ys, "o-", color="#17365D"); ax[3].axvline(t, ls="--", color="#A61C1C")
ax[3].set_xlabel("threshold"); ax[3].set_ylabel("components"); ax[3].set_title("count vs threshold", fontsize=13)
for a in ax[:3]: a.axis("off")
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T02_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T02_panel.png")

# ===== SUPPORT cell (run with: python3 T02_studio.py support) ==============
# Pre-wired: fixed threshold 104, mask ready. Do steps 2 and 4 only.
def support():
    b = (cv2.GaussianBlur(img, (5, 5), 1) > 104).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n8, _ = cv2.connectedComponents(b, connectivity=8)
    print(f"[SUPPORT] components (8-adj): {n8-1}. Now change 104 to 84 and to 124, rerun, and write down what happens.")

# ===== EXTENSION cell (run with: python3 T02_studio.py extension) ==========
# m-adjacency on a staircase WITH a bridge pixel: 8-adjacency sees duplicate
# paths between some pixel pairs; m-adjacency admits a diagonal link ONLY if
# the two pixels share no 4-neighbour. Count components under all three rules.
def extension():
    g = np.eye(4, dtype=np.uint8); g[1, 0] = 1        # staircase + bridge
    n4, _ = cv2.connectedComponents(g, connectivity=4)
    n8, _ = cv2.connectedComponents(g, connectivity=8)
    # m-adjacency by hand (union-find over admitted links)
    idx = {tuple(p): i for i, p in enumerate(np.argwhere(g))}
    parent = list(range(len(idx)))
    def find(a):
        while parent[a] != a: a = parent[a]
        return a
    def union(a, b): parent[find(a)] = find(b)
    for (y, x), i in idx.items():
        for dy, dx in ((0,1),(1,0),(0,-1),(-1,0)):
            if (y+dy, x+dx) in idx: union(i, idx[(y+dy, x+dx)])
        for dy, dx in ((1,1),(1,-1),(-1,1),(-1,-1)):
            q = (y+dy, x+dx)
            if q in idx:
                # shared 4-neighbour = the two corner cells of the diagonal step
                s1, s2 = (y, x+dx), (y+dy, x)
                if not (s1 in idx or s2 in idx): union(i, idx[q])
    nm = len({find(i) for i in range(len(idx))})
    print(f"[EXTENSION] bridged staircase: 4-adj {n4-1} | 8-adj {n8-1} | m-adj {nm} components")
    print("             In two lines: why does m-adjacency refuse the diagonal link where a bridge exists?")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()