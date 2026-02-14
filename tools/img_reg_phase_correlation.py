import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rotate, rescale
from scipy.ndimage import fourier_shift
from lib.phase_corr_functions import phase_corr_translation, phase_corr_translation_rotation_scaling
import logging

logging.basicConfig(level=logging.INFO)

img_A_path = "../img_data/NewData/IMG_0000_1.tif"
img_A = cv.imread(img_A_path, cv.IMREAD_COLOR)
# img_A = cv.imread(img_A_path, cv.IMREAD_GRAYSCALE)
img_B_path = "../img_data/NewData/IMG_0000_2.tif"
img_B = cv.imread(img_B_path, cv.IMREAD_COLOR)
# img_B = cv.imread(img_B_path, cv.IMREAD_GRAYSCALE)
img_C_path = "../img_data/NewData/IMG_0000_3.tif"
img_C = cv.imread(img_C_path, cv.IMREAD_COLOR)
# img_C = cv.imread(img_C_path, cv.IMREAD_GRAYSCALE)
img_D_path = "../img_data/NewData/IMG_0000_4.tif"
img_D = cv.imread(img_D_path, cv.IMREAD_COLOR)
# img_D = cv.imread(img_D_path, cv.IMREAD_GRAYSCALE)


img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
img_C_gray = cv.cvtColor(img_C, cv.COLOR_BGR2GRAY)
img_D_gray = cv.cvtColor(img_D, cv.COLOR_BGR2GRAY)

# img_A_gray = img_A
# img_B_gray = img_B
# img_C_gray = img_C
# img_D_gray = img_D

# # img_B_gray = rotate(img_A_gray, 38.7)
# img_B_gray = rescale(img_B_gray, 2.3)

radius = 1000

# shift = (-22.4, -14.89)
# offset_image = fourier_shift(np.fft.fft2(img_B_gray), shift)
# img_B_gray = np.abs(np.fft.ifft2(offset_image))

# plt.imshow(img_A_gray, cmap='gray')
# plt.show()
# plt.imshow(img_B_gray, cmap='gray')
# plt.show()

sigma = min(img_A_gray.shape) * 0.3

# img_translation = phase_corr_translation_rotation_scaling(img_A_gray, img_B_gray, sigma)
# img_translation_2 = phase_corr_translation_rotation_scaling(img_A_gray, img_C_gray, sigma)
# img_translation_3 = phase_corr_translation_rotation_scaling(img_A_gray, img_D_gray, sigma)

img_translation = phase_corr_translation(img_A_gray, img_B_gray, sigma)
img_translation_2 = phase_corr_translation(img_A_gray, img_C_gray, sigma)
img_translation_3 = phase_corr_translation(img_A_gray, img_D_gray, sigma)

final_img = np.zeros((img_A_gray.shape[0], img_A_gray.shape[1], 3), dtype=np.uint8)
final_img[:, :, 0] = img_translation
# final_img[:, :, 0] = img_A_gray
final_img[:, :, 1] = cv.cvtColor(img_translation_2, cv.COLOR_GRAY2BGR)
final_img[:, :, 2] = cv.cvtColor(img_translation_3, cv.COLOR_GRAY2BGR)

logging.info(f"IMG_MATRIX {final_img}")

for i in range(3):
    plt.imshow(final_img[:, :, i])
    plt.show()

plt.imshow(final_img)
plt.show()
