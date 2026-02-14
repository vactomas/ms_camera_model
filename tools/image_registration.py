'''
=======================================================================================================================
- Name:         Image registration
- Description:  Takes images from multispectral camera and combines it into one
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.transform import PiecewiseAffineTransform, ThinPlateSplineTransform, warp
import sys

## Images
# img_ref_path = '../image_data/0002SET/000/IMG_0000_1.tif'
# img_src_path = '../image_data/0002SET/000/IMG_0000_2.tif'
# img_src_path_2 = '../image_data/0002SET/000/IMG_0000_3.tif'
img_ref_path = '../image_data/Phone/1.jpg'
img_src_path = '../image_data/Phone/2.jpg'

## Masks
mask_ref = None
mask_src = None

# Num of good matches
N_GOOD_MATCHES = 50

img_ref = cv2.imread(img_ref_path, cv2.IMREAD_GRAYSCALE)
img_src = cv2.imread(img_src_path, cv2.IMREAD_GRAYSCALE)
# img_src_2 = cv2.imread(img_src_path_2, cv2.IMREAD_GRAYSCALE)

# finder = cv2.SIFT_create()
finder = cv2.ORB_create()
# finder = cv2.FastFeatureDetector_create()
kp_ref, des_ref = finder.detectAndCompute(img_ref, mask_ref)
# kp_ref = finder.detect(img_ref, mask_ref)
kp_src, des_src = finder.detectAndCompute(img_src, mask_src)
# kp_src_2, des_src_2 = finder.detectAndCompute(img_src_2, mask_src)

# img_ref_with_kp = cv2.drawKeypoints(img_ref, kp_ref, 0, (0, 255, 0),
                                 # flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = matcher.match(des_ref, des_src)
# matches_2 = matcher.match(des_ref, des_src_2)

good_matches = matches[:N_GOOD_MATCHES]
# good_matches_2 = matches_2[:N_GOOD_MATCHES]

img_matches = cv2.drawMatches(img_ref, kp_ref, img_src, kp_src, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 5))
plt.imshow(img_matches)
plt.title('Feature Matches')
plt.show()

# sys.exit(1)

pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])
# pts_ref_2 = np.float32([kp_ref[m.queryIdx].pt for m in good_matches_2]).reshape(-1, 1, 2)

# Should be kp_src and kp_src_2 I guess but it doesn't work with it for some reason
# It only works if I put kp_ref in there and then it does some funky stuff and in reality doesn't work either
pts_src = np.float32([kp_src[m.queryIdx].pt for m in good_matches])
# pts_src_2 = np.float32([kp_src_2[m.queryIdx].pt for m in good_matches_2]).reshape(-1, 1, 2)

src_coords = pts_src.squeeze()
# src_coords_2 = pts_src_2.squeeze()
ref_coords = pts_ref.squeeze()
# ref_coords_2 = pts_ref_2.squeeze()

# tform = PiecewiseAffineTransform()
# tform = ThinPlateSplineTransform()
# tform_2 = tform
# tform.estimate(src_coords, ref_coords)
# tform_2.estimate(src_coords_2, ref_coords_2)

# # Underneath is switch coords experiment, it doesn't work
# tform.estimate(ref_coords, src_coords)
# tform_2.estimate(ref_coords_2, src_coords_2)

matrix = cv2.getPerspectiveTransform(pts_src, pts_ref)
# matix = cv2.getPerspectiveTransform(src_coords, ref_coords)

warped_image = cv2.warpPerspective(img_src, matrix, (img_ref.shape[1], img_ref.shape[0]))

# img_warped = warp(img_src, tform, output_shape=img_ref.shape)
# img_warped_2 = warp(img_src_2, tform_2, output_shape=img_ref.shape)

final_img = np.zeros((img_ref.shape[0], img_ref.shape[1], 3))
final_img[:, :, 0] = img_ref
# final_img[:, :, 1] = img_warped
final_img[:, :, 1] = warped_image
# final_img[:, :, 2] = img_warped_2

plot_data = final_img
plot_data *= (1.0 / plot_data.max())

plt.imshow(plot_data)
# plt.imshow(img_ref_with_kp)
# plt.imshow(img_warped_2)

plt.show()

print(f"REF: {final_img[:, :, 0]}")
print(f"SRC: {final_img[:, :, 1]}")
