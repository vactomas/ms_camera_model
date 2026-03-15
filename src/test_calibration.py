import logging
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from ms_camera_model import (
    AreaLocation,
    DataComparator,
    FilterSensorUnit,
    HyperspectralImageData,
    MultispectralCameraModel,
    MultispectralImageData,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] -> %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S%p",
                    handlers=[logging.FileHandler(f"logfile.log"),
                              logging.StreamHandler(sys.stdout)])

# Paths
hs_path = "../img_data/Multicam_0915-1416.hdr"
ms_paths = [
    "../img_data/0003SET/000/IMG_0000_1.tif", "../img_data/0003SET/000/IMG_0000_2.tif",
    "../img_data/0003SET/000/IMG_0000_3.tif", "../img_data/0003SET/000/IMG_0000_4.tif",
    "../img_data/0003SET/000/IMG_0000_5.tif"
]

filter_paths = [
    "../filters_sensors/AltumPT_Blue_RSR_SG.xlsx", "../filters_sensors/AltumPT_Green_RSR_SG.xlsx",
    "../filters_sensors/AltumPT_Red_RSR_SG.xlsx", "../filters_sensors/AltumPT_Red_Edge_RSR_SG.xlsx",
    "../filters_sensors/AltumPT_NIR_RSR_SG.xlsx"
]

sensor_paths = [
    "../filters_sensors/Sony_CMOS_IMX_Pregius_Gen_2.xlsx", "../filters_sensors/Sony_CMOS_IMX_Pregius_Gen_2.xlsx",
    "../filters_sensors/Sony_CMOS_IMX_Pregius_Gen_2.xlsx", "../filters_sensors/Sony_CMOS_IMX_Pregius_Gen_2.xlsx",
    "../filters_sensors/Sony_CMOS_IMX_Pregius_Gen_2.xlsx"
]

band_names = ["Blue", "Green", "Red", "Red edge", "NIR"]
panel_calibration = {"Blue": 51.2, "Green": 51.3, "Red": 51.3, "Red edge": 51.2, "NIR": 51.0}
panel_data_filepath = "../img_data/serial_data_RP06-2315006-OB.csv"

ms_panel_location = [
    AreaLocation(1294, 739, 1363, 791),
    AreaLocation(1338, 710, 1390, 760),
    AreaLocation(1344, 747, 1394, 797),
    AreaLocation(1327, 724, 1377, 774),
    AreaLocation(1317, 753, 1367, 803)
]

hs_panel_location = AreaLocation(149, 607, 213, 658)

ms_data = MultispectralImageData.import_altum_pt_ms_imgs(ms_paths, panel_calibration, ms_panel_location)
ms_data.normalize_img_data()

hs_data = HyperspectralImageData.import_calibrated_hs_img(hs_path, panel_data_filepath, hs_panel_location)
hs_data.normalize_img_data()

fs_units = []

for i, _ in enumerate(ms_paths):
    fs_unit = FilterSensorUnit.from_excel(filter_paths[i], sensor_paths[i])
    fs_units.append(fs_unit)

ms_cam_model = MultispectralCameraModel(hs_data, fs_units, band_names)
ms_cam_model.run_simulation()

data_comparator = DataComparator(ms_data, ms_cam_model.out_data)

real_ms_area_location = [
    AreaLocation(1170, 680, 1190, 700),
    AreaLocation(1210, 660, 1230, 680),
    AreaLocation(1220, 685, 1240, 705),
    AreaLocation(1200, 665, 1220, 685),
    AreaLocation(1190, 695, 1210, 715)
]
hs_panel_location = AreaLocation(310, 480, 330, 500)

modeled_ms_area_location = [
    hs_panel_location, hs_panel_location, hs_panel_location, hs_panel_location, hs_panel_location
]

real_ratios, modeled_ratios = data_comparator.compare_band_ratios(real_ms_area_location,
                                                                  modeled_ms_area_location,
                                                                  set_areas_globally=False)

SAM_angle = data_comparator.calculate_spectral_angle_mapper(real_ratios, modeled_ratios)
logging.info(f"[MAIN] The SAM angle value is {SAM_angle}.")

w, x = 0.4, np.arange(len(real_ratios))
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(ms_data.img_data[:, :, (2, 1, 0)])
ax[0, 1].imshow(cv.normalize(ms_cam_model.out_data.img_data[:, :, (2, 1, 0)], None, 255, 0, cv.NORM_MINMAX, cv.CV_8U))
ax[1, 0].bar(x, real_ratios, width=w, label="Ratios of real MS data")
ax[1, 1].bar(x, modeled_ratios, width=w, label="Ratios of modeled MS data")

plt.show()
