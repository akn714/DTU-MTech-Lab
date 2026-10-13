"""AI531C T10 studio — Transfer Clinic + Design Review (Tue 13 Oct, 2-3 PM).
INPUT: your project's numbers (dataset size, domain similarity) + this calculator.
OPERATION: compute trainable fractions for three freeze strategies on a 1.2M-param
  backbone; place YOUR project in the freeze/fine-tune quadrant; peer design review.
PARAMETERS: backbone 1.2M params; head = 3x3 conv 64->2 with bias.
EXPECTED OUTPUT: the three strategies' trainable counts (asserted, incl. the L26
  1,154 / 0.10% figure).
INTERPRETATION: freezing turns a million-parameter problem into a thousand-parameter
  one — the quadrant decides how much to thaw.
"""
BACKBONE = 1_200_000
head = (3*3*64 + 1) * 2
print(f"[1] head parameters: (3*3*64+1)*2 = {head}")
assert head == 1154

strategies = {
  "freeze all, train head":            head,
  "unfreeze last block (~15%) + head": int(0.15*BACKBONE) + head,
  "fine-tune everything":              BACKBONE + head,
}
for s, n in strategies.items():
    print(f"[2] {s:36s} trainable {n:9,d}  ({n/(BACKBONE+head):6.2%})")
assert abs(head/(BACKBONE+head) - 0.00096) < 0.0002

print("""[3] THE QUADRANT — place your project (write the cell letter on your sheet):
              domain SIMILAR      domain FAR
  data SMALL      A: freeze+head     B: freeze early only / classical features
  data LARGE      C: fine-tune all   D: fine-tune from early ckpt / from scratch
[4] DESIGN REVIEW (pairs): exchange one-page architecture sketches; reviewer checks:
    shapes close? RF covers the largest object (T09's calculator)? quadrant defensible?""")

def support():
    print("[SUPPORT] compute only the head for YOUR class count C: (3*3*64+1)*C.")
    print("  For C = 5:", (3*3*64+1)*5, "- then place your project in the quadrant with your TA.")

def extension():
    print("[EXTENSION] progressive unfreezing schedule: write the trainable-parameter count")
    print("  after each of 4 stages (head; +25%; +50%; all) and sketch count-vs-stage.")
    for f in (0, .25, .5, 1.0): print(f"   stage {f:.0%}: {int(f*BACKBONE)+1154:9,d}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "support": support()
    elif len(sys.argv) > 1 and sys.argv[1] == "extension": extension()