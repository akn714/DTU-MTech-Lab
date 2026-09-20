"""AI531C T06 studio — Registration Lab (Tue 8 Sep, 2-3 PM).
INPUT: camera as the reference; a synthetically rotated(+7 deg)+shifted(12,-8) copy
  as the 'second date' — so ground truth is KNOWN.
OPERATION: recover the affine from 3 clean point pairs -> measure residual ->
  inject ONE corrupted pair (30 px off) into a 6-pair least-squares fit (watch it
  bend) -> rescue with RANSAC via estimateAffinePartial2D.
PARAMETERS: angle 7 deg; shift (12, -8); outlier offset 30 px; RANSAC thr 3.
EXPECTED OUTPUT: printed recoveries (asserted) + expected_outputs/T06_panel.png.
INTERPRETATION: least squares follows outliers (squared error!); RANSAC votes them out.
"""
import numpy as np, cv2, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import data
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/expected_outputs", exist_ok=True); os.makedirs(f"{HERE}/inputs", exist_ok=True)
img = data.camera()
Mtrue = cv2.getRotationMatrix2D((256, 256), 7, 1.0); Mtrue[:, 2] += (12, -8)
moved = cv2.warpAffine(img, Mtrue, (512, 512))
cv2.imwrite(f"{HERE}/inputs/reference.png", img); cv2.imwrite(f"{HERE}/inputs/moved.png", moved)

pts = np.array([[100,100],[400,120],[250,400],[120,380],[420,380],[256,200]], np.float64)
dst = (Mtrue[:, :2] @ pts.T).T + Mtrue[:, 2]

# [1] exact recovery from 3 pairs
M3 = cv2.getAffineTransform(pts[:3].astype(np.float32), dst[:3].astype(np.float32))
ang = np.degrees(np.arctan2(M3[1,0], M3[0,0]))
print(f"[1] 3-pair recovery: angle {ang:.2f} deg (truth 7.00), tx {M3[0,2]:.1f} ty {M3[1,2]:.1f}")
assert abs(ang - (-7)) < 0.05 or abs(ang - 7) < 0.05

# [2] one bad pair poisons PLAIN least squares (fit it ourselves, transparently)
bad = dst.copy(); bad[5] += (30, -30)
A = np.zeros((12, 6)); A[0::2, 0:2] = pts; A[0::2, 2] = 1; A[1::2, 3:5] = pts; A[1::2, 5] = 1
M6 = np.linalg.lstsq(A, bad.ravel(), rcond=None)[0].reshape(2, 3)
res6 = np.abs((M6[:, :2] @ pts.T).T + M6[:, 2] - dst).max()
print(f"[2] 6 pairs, one 30-px outlier, plain least squares: worst residual vs truth {res6:.1f} px")
assert res6 > 5   # the outlier bent the whole fit

# [3] RANSAC rescue
Mr, mask = cv2.estimateAffine2D(pts.astype(np.float32), bad.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=3.0)
resR = np.abs((Mr[:, :2] @ pts.T).T + Mr[:, 2] - dst).max()
print(f"[3] RANSAC (thr 3): kept {int(mask.sum())}/6 pairs, worst residual {resR:.2f} px")
assert int(mask.sum()) == 5 and resR < 1.0

back = cv2.warpAffine(moved, cv2.invertAffineTransform(Mr), (512, 512))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, im, t in zip(ax, [img, moved, np.abs(img.astype(int)-moved.astype(int)), np.abs(img.astype(int)-back.astype(int))],
    ["reference", "moved (+7 deg, shift)", "|difference| BEFORE", "|difference| AFTER RANSAC"]):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=13); a.axis("off")
plt.tight_layout(); plt.savefig(f"{HERE}/expected_outputs/T06_panel.png", dpi=120); plt.close()
print("saved expected_outputs/T06_panel.png")

def support():
    print("[SUPPORT] use only step [1]: three clean pairs, cv2.getAffineTransform.")
    print("  Verify the recovered angle is ~7 deg. Then change ONE point by 5 px and watch the angle move.")

def extension():
    h = np.eye(3); h[2, 0] = 2e-4   # mild perspective
    pers = cv2.warpPerspective(img, h @ np.vstack([Mtrue, [0,0,1]]), (512, 512))
    cv2.imwrite(f"{HERE}/inputs/perspective.png", pers)
    print("[EXTENSION] inputs/perspective.png has mild PERSPECTIVE. Show the best affine fit")
    print("  cannot flatten the residual, then recover it with cv2.findHomography (4+ pairs).")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()