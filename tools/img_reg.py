

from __future__ import annotations
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

N_GOOD_MATCHES = 50

# Paths
img_ref_path = "../image_data/Phone/1.jpg"
img_src_path = "../image_data/Phone/4.jpg"
img_src_2_path = "../image_data/Phone/3.jpg"
# img_ref_path = "../image_data/0002SET/000/IMG_0000_1.tif"
# img_src_path = "../image_data/0002SET/000/IMG_0000_2.tif"
# img_src_2_path = "../image_data/0002SET/000/IMG_0000_3.tif"

# Masks
mask_ref = None
mask_src = None

# Load images
img_ref = cv.imread(img_ref_path, cv.IMREAD_COLOR)
img_src = cv.imread(img_src_path, cv.IMREAD_COLOR)
img_src_2 = cv.imread(img_src_2_path, cv.IMREAD_COLOR)

# Make them grayscale
img_ref_gray = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)
img_src_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
img_src_2_gray = cv.cvtColor(img_src_2, cv.COLOR_BGR2GRAY)

# Find key points
finder = cv.ORB_create()
kp_ref, des_ref = finder.detectAndCompute(img_ref_gray, mask_ref)
kp_src, des_src = finder.detectAndCompute(img_src_gray, mask_src)
kp_src_2, des_src_2 = finder.detectAndCompute(img_src_2_gray, mask_src)

# Match
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des_ref, des_src)
matches_2 = matcher.match(des_ref, des_src_2)
matches = sorted(matches, key = lambda x:x.distance)
matches_2 = sorted(matches_2, key = lambda y:y.distance)

matches = matches[:int(len(matches)*90)]
matches_2 = matches_2[:int(len(matches_2)*90)]
print(f"Num of good matches: {len(matches)}")
print(f"Num of good matches img_2: {len(matches_2)}")

p1 = np.zeros((len(matches), 2))
p2 = np.zeros((len(matches), 2))
p1_2 = np.zeros((len(matches_2), 2))
p2_2 = np.zeros((len(matches_2), 2))

for i in range(len(matches)):
    p1[i, :] = kp_ref[matches[i].queryIdx].pt
    p2[i, :] = kp_src[matches[i].trainIdx].pt

for i in range(len(matches_2)):
    p1_2[i, :] = kp_ref[matches_2[i].queryIdx].pt
    p2_2[i, :] = kp_src_2[matches_2[i].trainIdx].pt

homography, mash = cv.findHomography(p2, p1, cv.RANSAC)
homography_2, mash_2 = cv.findHomography(p2_2, p1_2, cv.RANSAC)

transformed_img = cv.warpPerspective(img_src, homography, (img_ref.shape[1], img_ref.shape[0]))
transformed_img_2 = cv.warpPerspective(img_src_2, homography_2, (img_ref.shape[1], img_ref.shape[0]))

final_img = np.zeros((img_ref.shape[0], img_ref.shape[1], 3))
final_img[:, :, 0] = img_ref_gray
final_img[:, :, 1] = cv.cvtColor(transformed_img, cv.COLOR_BGR2GRAY)
final_img[:, :, 2] = cv.cvtColor(transformed_img_2, cv.COLOR_BGR2GRAY)

# plt.imshow(cv.cvtColor(transformed_img, cv.COLOR_BGR2GRAY))
print(final_img)
print(final_img.shape)

plt.imshow(final_img.astype(int))
plt.show()
