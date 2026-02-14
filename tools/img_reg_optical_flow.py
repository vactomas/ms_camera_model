import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import cv2 as cv

# --- Convert the images to gray level: color is not supported.
img_A_path = "../img_data/NewData/IMG_0000_1.tif"
img_B_path = "../img_data/NewData/IMG_0000_2.tif"
img_C_path = "../img_data/NewData/IMG_0000_3.tif"
img_D_path = "../img_data/NewData/IMG_0000_4.tif"

image0 = cv.imread(img_A_path, cv.IMREAD_GRAYSCALE)
image1 = cv.imread(img_B_path, cv.IMREAD_GRAYSCALE)
image2 = cv.imread(img_C_path, cv.IMREAD_GRAYSCALE)
image3 = cv.imread(img_D_path, cv.IMREAD_GRAYSCALE)

# --- Compute the optical flow
v, u = optical_flow_tvl1(image0, image1)
v2, u2 = optical_flow_tvl1(image0, image2)
v3, u3 = optical_flow_tvl1(image0, image3)

# --- Use the estimated optical flow for registration

nr, nc = image0.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='edge')
image2_warp = warp(image2, np.array([row_coords + v, col_coords + u]), mode='edge')
image3_warp = warp(image3, np.array([row_coords + v, col_coords + u]), mode='edge')

# build an RGB image with the unregistered sequence
seq_im = np.zeros((nr, nc, 3), dtype=np.uint8)
seq_im[..., 0] = image1
seq_im[..., 1] = image2
seq_im[..., 2] = image3

# build an RGB image with the registered sequence
reg_im = np.zeros((nr, nc, 3), dtype=np.float32)
reg_im[..., 0] = image1_warp
reg_im[..., 1] = image2_warp
reg_im[..., 2] = image3_warp

# build an RGB image with the registered sequence
target_im = np.zeros((nr, nc, 3), dtype=np.uint8)
target_im[..., 0] = image0
target_im[..., 1] = image0
target_im[..., 2] = image0

# --- Show the result

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))

ax0.imshow(seq_im)
ax0.set_title("Unregistered sequence")
ax0.set_axis_off()

ax1.imshow(reg_im)
ax1.set_title("Registered sequence")
ax1.set_axis_off()

ax2.imshow(target_im)
ax2.set_title("Target")
ax2.set_axis_off()

fig.tight_layout()
plt.show()
