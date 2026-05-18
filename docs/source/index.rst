.. Multispectral Camera Simulator documentation master file, created by
   sphinx-quickstart on Sat Feb 28 19:55:17 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Multispectral Camera Simulator documentation
============================================

This Python package offers a simple way of simulating a response of multispectral camera based on hyperspectral input.

Source code is available on GitHub: https://github.com/vactomas/ms_camera_model

Package is available on PyPI under the name `ms_camera_model`. To install it, run `pip install ms_camera_model`.

An example of such simulation is defined below.

.. code:: python

    import logging
    import sys

    import matplotlib.pyplot as plt

    from ms_camera_model import (
        FilterSensorUnit,
        HyperspectralImageData,
        MultispectralCameraModel,
    )
    from ms_camera_model.image_visualiser import imshow

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] -> %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S%p",
                        handlers=[logging.StreamHandler(sys.stdout)])

    filter_paths = ["path/to/filter.xlsx"]

    sensor_paths = ["path/to/sensor.xlsx"]

    hs_path = "path/to/hyperspectral.hdr"
    hs_data = HyperspectralImageData.import_hs_img(hs_path)

    band_names = ["band1"]

    fs_units = []

    for i in range(len(filter_paths)):
        fs_unit = FilterSensorUnit.from_excel(filter_paths[i], sensor_paths[i])
        fs_units.append(fs_unit)

    ms_cam_model = MultispectralCameraModel.create_model(hs_data, fs_units, band_names)
    ms_cam_model.run_simulation()

    imshow(ms_cam_model.out_data, [0])
    plt.show()

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   changelog
