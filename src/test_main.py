import logging
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] -> %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S%p",
                    handlers=[
                        logging.FileHandler(f"logfile.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

from ms_camera_model import *

# Paths
hs_path = "../img_data/Multicam_0915-1416.hdr"
ms_paths = [
    "../img_data/0003SET/000/IMG_0000_1.tif",
    "../img_data/0003SET/000/IMG_0000_2.tif",
    "../img_data/0003SET/000/IMG_0000_3.tif",
    "../img_data/0003SET/000/IMG_0000_4.tif",
    "../img_data/0003SET/000/IMG_0000_5.tif"
]
# masks_path_src = ["../img_data/0003SET/000/IMG_0000_1_mask.tif", "../img_data/0003SET/000/IMG_0000_2_mask.tif", "../img_data/0003SET/000/IMG_0000_3_mask.tif", "../img_data/0003SET/000/IMG_0000_4_mask.tif", "../img_data/0003SET/000/IMG_0000_5_mask.tif"]
# masks_path_ref = ["../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif", "../img_data/Multicam_0915-1416_mask.tif"]
filter_paths = [
    "../filters_sensors/filter_mica_475_32.xlsx",
    "../filters_sensors/filter_mica_560_27.xlsx",
    "../filters_sensors/filter_mica_668_16.xlsx",
    "../filters_sensors/filter_mica_717_12.xlsx",
    "../filters_sensors/filter_mica_842_57.xlsx"
]
sensor_paths = [
    "../filters_sensors/multispectral_sensor_mock.xlsx",
    "../filters_sensors/multispectral_sensor_mock.xlsx",
    "../filters_sensors/multispectral_sensor_mock.xlsx",
    "../filters_sensors/multispectral_sensor_mock.xlsx",
    "../filters_sensors/multispectral_sensor_mock.xlsx"
]

panel_calibration = {
    "Blue": 51.2,
    "Green": 51.3,
    "Red": 51.3,
    "Red edge": 51.2,
    "NIR": 51.0
}

panel_location = [[1294, 739, 1363, 791], [1338, 710, 1390, 760],
                  [1344, 747, 1394, 797], [1327, 724, 1377, 774],
                  [1317, 753, 1367, 803]]

ms_data = MultispectralImageData()
ms_data.import_altum_pt_ms_imgs(ms_paths, panel_calibration, panel_location)
ms_data.normalize_img_data()

hs_data = HyperspectralImageData()
hs_data.import_hs_img(hs_path)
hs_data.normalize_img_data()

fs_units = []

for i in range(len(ms_paths)):
    fs_unit = FilterSensorUnit()
    fs_unit.load_filter_sensor(filter_paths[i], sensor_paths[i])
    fs_units.append(fs_unit)

ms_cam_model = MultispectralCameraModel(hs_data, fs_units)

data_comparator = DataComparator(ms_data, ms_cam_model.out_data)
real_ratios, modeled_ratios = data_comparator.compare_band_ratios(
    set_areas_globaly=False)

w, x = 0.4, np.arange(len(real_ratios))
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(ms_data.img_data[:, :, (2, 1, 0)])
ax[0, 1].imshow(
    cv.normalize(ms_cam_model.out_data.img_data[:, :, (2, 1, 0)], None, 255, 0,
                 cv.NORM_MINMAX, cv.CV_8U))
ax[1, 0].bar(x, real_ratios, width=w, label="Ratios of real MS data")
ax[1, 1].bar(x, modeled_ratios, width=w, label="Ratios of modeled MS data")

plt.show()
