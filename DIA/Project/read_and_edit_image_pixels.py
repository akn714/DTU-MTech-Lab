from PIL import Image
import numpy as np

# Load image
img = Image.open("input.jpeg").convert("RGB")

# Convert image to N x M x 3 pixel grid
pixels = np.array(img)

print("Image shape:", pixels.shape)
print("Pixel at (10, 20):", pixels[10, 20])

# -------------------------------------------------
# EDIT PIXELS HERE
# -------------------------------------------------

""" ---- Negative Transformation ---- """
# This is Negative Transformation of the image
for i in range(0, pixels.shape[0]):
    for j in range(0, pixels.shape[1]):
        pixels[i, j] = [255 - pixels[i, j][0], 255 - pixels[i, j][1], 255 - pixels[i, j][2]]

""" ---- Log Transformation ---- """
# This is Log Transformation of the image
# pixels_float = pixels.astype(np.float64)
# for i in range(pixels.shape[0]):
#     for j in range(pixels.shape[1]):
#         pixels[i, j] = [
#             int(255 * np.log(1 + pixels_float[i, j, 0]) / np.log(256)),
#             int(255 * np.log(1 + pixels_float[i, j, 1]) / np.log(256)),
#             int(255 * np.log(1 + pixels_float[i, j, 2]) / np.log(256))
#         ]

""" ---- Making some pixels green ---- """
# for i in range(100, 200):
#     for j in range(0, 100):
#         pixels[i, j] = [0, 255, 0]

""" ---- Making some pixels red ---- """
# for i in range(200, 300):
#     for j in range(0, 100):
#         pixels[i, j] = [255, 0, 0]

""" ---- Making some pixels black ---- """
# for i in range(300, 400):
#     for j in range(0, 100):
#         pixels[i, j] = [0, 0, 0]

""" ---- Making some pixels white ---- """
# pixels[100:110, 100:110] = [255, 255, 255]

# -------------------------------------------------

# Convert pixel grid back to image
output = Image.fromarray(pixels)

# Save
output.save("output.png")

print("Saved edited image as output.png")
