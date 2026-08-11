import cv2
import numpy as np

g = np.eye(4, dtype=np.uint8)          # the staircase
# B = (img > 128).astype(np.uint8)  # binary image
# 128 are number of pixels
n4, _ = cv2.connectedComponents(g, connectivity=4)
n8, _ = cv2.connectedComponents(g, connectivity=8)
print(n4 - 1, n8 - 1)                  # why the '- 1'? because the first label is for the background, so we subtract 1 to get the number of foreground components.
