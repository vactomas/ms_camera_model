import sys
import logging
import spectral
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%p",
    handlers=[logging.FileHandler(f"logfile.log"),
              logging.StreamHandler(sys.stdout)]
)

from ms_camera_model import *

# Paths
# hs_path = "../img_data/Multicam_0915-1416.hdr"
# ms_paths = ["../img_data/0003SET/000/IMG_0000_1.tif", "../img_data/0003SET/000/IMG_0000_2.tif", "../img_data/0003SET/000/IMG_0000_3.tif", "../img_data/0003SET/000/IMG_0000_4.tif", "../img_data/0003SET/000/IMG_0000_5.tif"]
# masks_path_src = ["../img_data/0003SET/000/IMG_0000_1_mask.tif", "../img_data/0003SET/000/IMG_0000_2_mask.tif", "../img_data/0003SET/000/IMG_0000_3_mask.tif", "../img_data/0003SET/000/IMG_0000_4_mask.tif", "../img_data/0003SET/000/IMG_0000_5_mask.tif"]
# masks_path_ref = ["../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif"]
# filter_paths = ["../filters_sensors/filter_mica_475_32.xlsx", "../filters_sensors/filter_mica_560_27.xlsx", "../filters_sensors/filter_mica_668_16.xlsx", "../filters_sensors/filter_mica_717_12.xlsx", "../filters_sensors/filter_mica_842_57.xlsx"]
# sensor_paths = ["../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx"]

# hs_path = "../img_data/NewData/plastak_1020-0308.hdr"
# hs_path = "../img_data/NewData/plastak_1020-0224.hdr"
# ms_paths = ["../img_data/NewData/IMG_0000_1.tif", "../img_data/NewData/IMG_0000_2.tif", "../img_data/NewData/IMG_0000_3.tif", "../img_data/NewData/IMG_0000_4.tif"]
# masks_path_src = ["../img_data/NewData/IMG_0000_1_mask.tif", "../img_data/NewData/IMG_0000_2_mask.tif", "../img_data/NewData/IMG_0000_3_mask.tif", "../img_data/NewData/IMG_0000_4_mask.tif"]
# masks_path_ref = ["../img_data/NewData/mask.tif", "../img_data/NewData/mask.tif", "../img_data/NewData/mask.tif", "../img_data/NewData/mask.tif", "../img_data/NewData/mask.tif"]
# masks_path_ref = [None, None, None, None]

hs_path = "../img_data/sumavska/20251211-hyperspectral/sumak_1211-1219.hdr"
ms_paths = ["../img_data/sumavska/0005SET/000/IMG_0034_1.tif", "../img_data/sumavska/0005SET/000/IMG_0034_2.tif", "../img_data/sumavska/0005SET/000/IMG_0034_3.tif", "../img_data/sumavska/0005SET/000/IMG_0034_4.tif"]
# masks_path_src = [None, None, None, None]
masks_path_src = ["../img_data/sumavska/0005SET/000/mask_ms.tif", "../img_data/sumavska/0005SET/000/mask_ms.tif", "../img_data/sumavska/0005SET/000/mask_ms.tif", "../img_data/sumavska/0005SET/000/mask_ms.tif"]
masks_path_ref = [None, None, None, None]
# filter_paths = ["../filters_sensors/filter_mica_475_32.xlsx", "../filters_sensors/filter_mica_560_27.xlsx", "../filters_sensors/filter_mica_668_16.xlsx", "../filters_sensors/filter_mica_717_12.xlsx"]
filter_paths = ["../filters_sensors/AltumPT_Blue_RSR.xlsx", "../filters_sensors/AltumPT_Green_RSR.xlsx", "../filters_sensors/AltumPT_Red_RSR.xlsx", "../filters_sensors/AltumPT_Red_Edge_RSR.xlsx"]
sensor_paths = ["../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx", "../filters_sensors/multispectral_sensor_mock.xlsx"]

hs_data = HyperspectralImageData()
hs_data.import_hs_img(hs_path)
hs_data.normalize_img_data()
# hs_data.vector_normalize()
hs_data.band_centers = spectral.open_image("../img_data/NewData/plastak_1020-0252.hdr").bands.centers

ms_data = MultispectralImageData()
ms_data.import_ms_imgs(ms_paths)
ms_data.normalize_img_data()
# ms_data.vector_normalize()

logging.info(f"[MAIN] MS data type: {ms_data.img_data.dtype}")
logging.info(f"[MAIN] HS data type: {hs_data.img_data.dtype}")
logging.info(f"[MAIN] MS data max value: {ms_data.img_data.max()}")
logging.info(f"[MAIN] HS data max value: {hs_data.img_data.max()}")

fs_units = []

for i in range(len(ms_paths)):
    fs_unit = FilterSensorUnit()
    fs_unit.load_filter_sensor(filter_paths[i], sensor_paths[i])
    fs_units.append(fs_unit)

ms_cam_model = MultispectralCameraModel(hs_data, fs_units)
logging.info(f"[MAIN] Modeled MS data type: {ms_cam_model.out_data.img_data.dtype}")
logging.info(f"[MAIN] Modeled MS data max value: {ms_cam_model.out_data.img_data.max()}")

# ms_cam_model = MultispectralCameraModel(HyperspectralImageData().import_hs_img(hs_path), fs_units)

# for i in range(ms_cam_model.out_data.nbands):
    # ms_cam_model.out_data.imshow([i])
# for i in range(ms_data.nbands):
#     ms_data.imshow([i])

# registered_real_ms_data = ms_cam_model.out_data.register_bands(ms_data, masks_path_ref, masks_path_src)
#
# registered_real_ms_data.imshow([3, 2, 1])
# registered_real_ms_data.imshow([2, 1, 0])

data_comparator = DataComparator(ms_data, ms_cam_model.out_data)
logging.info(f"[NOTE] If you haven't changed the data source, it is (400, 820, 20) for all modeled and for real ms (970, 850, 20), (990, 820, 20), (1000, 830, 20), (990, 830, 20)")
real_ratios, modeled_ratios = data_comparator.compare_band_ratios(set_areas_globaly = False)

w, x = 0.4, np.arange(len(real_ratios))
fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(registered_real_ms_data.img_data[:, :, (2, 1, 0)])
ax[0, 0].imshow(ms_data.img_data[:, :, (2, 1, 0)])
ax[0, 1].imshow(cv.normalize(ms_cam_model.out_data.img_data[:, :, (2, 1, 0)], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U))
ax[1, 0].bar(x, real_ratios, width=w, label="Ratios of real MS data")
ax[1, 1].bar(x, modeled_ratios, width=w, label="Ratios of modeled MS data")

plt.show()

input()

# plt.figure()
#
# for band in range(registered_real_ms_data.nbands):
#     plt.subplot(1, 4, band+1)
#     plt.imshow(registered_real_ms_data.img_data[:, :, band])
#
# plt.show()

# for band in range(registered_real_ms_data.nbands):
#     registered_real_ms_data.imshow([band])

# ms_cam_model.out_data.imshow([0, 1, 2])
