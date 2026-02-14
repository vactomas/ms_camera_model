from __future__ import annotations
import cv2 as cv
import numpy as np

import spectral

import matplotlib.pyplot as plt

from lib.dataclasses import ImageData

N_GOOD_MATCHES = 300

# Paths
img_hs_path = "../img_data/NewData/plastak_-3.hdr"
# img_ref_path = "../src/test.tif"
img_ref_path = "../img_data/NewData/IMG_0000_1.tif"
img_src_paths = ["../img_data/NewData/IMG_0000_2.tif", "../img_data/NewData/IMG_0000_3.tif", "../img_data/NewData/IMG_0000_4.tif", "../img_data/NewData/IMG_0000_5.tif", "../src/test.tif"]

# Masks
mask_src = []
# mask_src_paths = ["../img_data/NewData/IMG_0000_1_mask.tif", "../img_data/NewData/IMG_0000_2_mask.tif", "../img_data/NewData/IMG_0000_3_mask.tif", "../img_data/NewData/IMG_0000_4_mask.tif", "../img_data/NewData/IMG_0000_5_mask.tif"]
# mask_src_paths = ["../img_data/NewData/MS_mask_unified.tif", "../img_data/NewData/MS_mask_unified.tif", "../img_data/NewData/MS_mask_unified.tif", "../img_data/NewData/MS_mask_unified.tif", "../img_data/NewData/MS_mask_unified.tif"]
mask_src_paths = [None, None, None, None, None]
# mask_ref_path = "../img_data/NewData/mask.tif"

clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))

# Load images
# img_ref = spectral.open_image(img_hs_path).load().mean(axis=2)
# img_ref = cv.normalize(img_ref, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
img_ref = cv.imread(img_ref_path, cv.IMREAD_COLOR)
img_ref = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)
img_ref = clahe.apply(img_ref)


img_src_array = []
img_src_array_colour = []
num_of_images = len(img_src_paths)

for i_img in range(num_of_images):
    img = cv.imread(img_src_paths[i_img], cv.IMREAD_COLOR)
    img_src_array_colour.append(img)
    img = clahe.apply(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    img_src_array.append(img)

# Create masks
# mask_ref = cv.imread(mask_ref_path, cv.IMREAD_GRAYSCALE)
# _, mask_ref = cv.threshold(mask_ref, 1, 255, cv.THRESH_BINARY)
mask_ref = None
# mask_ref = np.zeros((img_ref.shape[0], img_ref.shape[1]), dtype='uint8')
# cv.rectangle(mask_ref, (475, 575), (100, 300), 255, -1)
# mask_ref = cv.bitwise_not(mask_ref)

for i_img in range(num_of_images):
    mask = cv.imread(mask_src_paths[i_img], cv.IMREAD_GRAYSCALE)
    ret, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    # mask = cv.bitwise_not(mask)
    mask_src.append(mask)

# Find key points
# finder = cv.ORB_create()
finder = cv.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=10) #, nOctaveLayers=5)
kp_ref, des_ref = finder.detectAndCompute(img_ref, mask_ref)

kp_src, des_src = [], []

for i_img in range(num_of_images):
    kp, des = finder.detectAndCompute(img_src_array[i_img], mask_src[i_img])
    kp_src.append(kp)
    des_src.append(des)
    print(f"Num of key point: {len(kp)}")

# Match
# matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches = []

for i_img in range(num_of_images):
    # tmp_matches = matcher.match(des_ref, des_src[i_img])
    tmp_matches = matcher.knnMatch(des_ref, des_src[i_img], k=2)
    # tmp_matches = sorted(tmp_matches, key = lambda x:x.distance)
    # tmp_matches = tmp_matches[:int(len(tmp_matches)*90)]
    good = []
    for m,n in tmp_matches:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    tmp_matches = good

    print(f"Num of good matches: {len(tmp_matches)}")

    img_matches = cv.drawMatches(img_ref, kp_ref, img_src_array[i_img], kp_src[i_img], tmp_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # img_matches = cv.drawMatchesKnn(img_ref, kp_ref, img_src_array[i_img], kp_src[i_img], tmp_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 5))
    plt.imshow(img_matches)
    plt.title('Feature Matches')
    plt.show()
    
    matches.append(tmp_matches)

final_img = np.zeros((img_ref.shape[0], img_ref.shape[1], num_of_images+1))
final_img[:, :, 0] = img_ref

for i_img in range(num_of_images):
    # p1 = np.zeros((len(matches[i_img]), 2))
    # p2 = np.zeros((len(matches[i_img]), 2))

    # for i_point in range(len(matches[i_img])):
        # p1[i_point, :] = kp_ref[matches[i_img][i_point].queryIdx].pt
        # p2[i_point, :] = kp_src[i_img][matches[i_img][i_point].trainIdx].pt
    
    p1 = np.float32([kp_ref[m.queryIdx].pt for m in matches[i_img]]).reshape(-1, 1, 2)
    p2 = np.float32([kp_src[i_img][m.trainIdx].pt for m in matches[i_img]]).reshape(-1, 1, 2)

    homography, _ = cv.findHomography(p2, p1, cv.RANSAC, 2.5, maxIters=20000)

    if homography is None:
        print(f"Couldn't find homography for image {i_img}, skipping...")
        continue

    try:
        transformed_img = cv.warpPerspective(img_src_array_colour[i_img], homography, (img_ref.shape[1], img_ref.shape[0]))
        final_img[:, :, i_img+1] = cv.cvtColor(transformed_img, cv.COLOR_BGR2GRAY)

    except Exception as ex:
        print(f"Failed due to:\n{ex}")

print(f"Final img cube shape {final_img.shape}")

img_data = ImageData(final_img, [], [])
img_data.imshow_rgb([1, 1, 1])
img_data.imshow_rgb([2, 2, 2])
img_data.imshow_rgb([3, 3, 3])
img_data.imshow_rgb([4, 4, 4])
img_data.imshow_rgb([5, 5, 5])
img_data.imshow_rgb([5, 1, 0])

