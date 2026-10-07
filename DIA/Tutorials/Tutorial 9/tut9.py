"""AI531C T09 studio — The Shape & RF Calculator (Tue 6 Oct, 2-3 PM).
INPUT: none (pure arithmetic, verified against real convolutions).
OPERATION: build two functions — out_size() and rf() — and CHECK them against
  actual cv2/scipy convolutions; then size an encoder for a target object.
PARAMETERS: the L22 trace (224 input) and the L23 recurrence.
EXPECTED OUTPUT: printed traces (asserted against scipy ground truth).
INTERPRETATION: two tiny functions replace guesswork for every architecture you read.
"""
import numpy as np
from scipy.signal import convolve2d

def out_size(n, k, s, p): return (n + 2*p - k)//s + 1
def rf_trace(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1)*jump; jump *= s
    return rf

# [1] verify out_size against a REAL convolution
x = np.zeros((224, 224))
y = convolve2d(x, np.ones((7, 7)), mode="valid")[::2, ::2]   # k7 s2 p0 (valid)
pred = out_size(224, 7, 2, 0)
print(f"[1] k7 s2 p0 on 224: scipy gives {y.shape[0]}, out_size() gives {pred}")
assert y.shape[0] == pred

# [2] the L22 trace, reproduced by YOUR function
trace = [(7,2,3),(3,2,1),(3,1,1),(3,2,1)]
n = 224
for k, s, p in trace:
    n = out_size(n, k, s, p); print(f"    k{k} s{s} p{p} -> {n}")
assert n == 28

# [3] receptive fields
for layers, name in [([(3,1)]*2, "two 3x3"), ([(3,1)]*3, "three 3x3"),
                     ([(3,1),(3,2),(3,1),(3,2),(3,1)], "with strides")]:
    print(f"[3] RF of {name}: {rf_trace(layers)}")
assert rf_trace([(3,1)]*2) == 5 and rf_trace([(3,1)]*3) == 7

# [4] design round: how deep until RF >= 100 (plain 3x3)? and with stride-2 every 2nd?
d = 0; layers = []
while rf_trace(layers) < 100: layers.append((3,1)); d += 1
print(f"[4] plain 3x3 layers for RF>=100: {d}")
layers2, d2 = [], 0
while rf_trace(layers2) < 100:
    layers2.append((3, 2 if d2 % 2 == 1 else 1)); d2 += 1
print(f"    alternating stride-2:        {d2} layers")
assert d2 < d

def support():
    print("[SUPPORT] use the two provided functions as black boxes; fill this table only:")
    print("  out_size(96,3,1,1)=?  out_size(96,3,2,1)=?  rf of four 3x3 layers=?")
    print("  answers:", out_size(96,3,1,1), out_size(96,3,2,1), rf_trace([(3,1)]*4))

def extension():
    def rf_dilated(layers):
        rf, jump = 1, 1
        for k, s, d in layers: rf += d*(k-1)*jump; jump *= s
        return rf
    print("[EXTENSION] add dilation to rf(): three 3x3 layers with dilation 1,2,4 ->",
          rf_dilated([(3,1,1),(3,1,2),(3,1,4)]), "(plain three 3x3 gave 7)")
    print("  Same 27 weights. Where is the catch? (Hint: gridding.)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()