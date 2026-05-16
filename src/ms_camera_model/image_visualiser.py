"""
Multispectral Camera Model - Image Visualisation
================================================

* **Description:** Functions for image visualisation
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from ms_camera_model.errors import (
    IncompatibleBandChoice,
    NoBandCenters,
)
from ms_camera_model.image_data import AreaLocation, ImageData

logger = logging.getLogger(__name__)


def imshow(image_data: ImageData, bands: list[int] | None = None) -> None:
    """ View image as an RGB interpretation of selected bands or as a brightness plot

    :param image_data: ImageData instance
    :param bands: list of band numbers, if the list is not provided, brightness plot will be used instead
    :raises TypeError: if provided bands are not a list or None
    :raises IncompatibleBandChoice: if 'bands' is not [], len(bands) != 3 or the number is out-of-bounds
    """

    if bands is None:
        bands = []

    if not isinstance(bands, list):
        raise TypeError(f"Expected list[int] or None, got {type(bands)}")

    if len(bands) == 3:
        logger.info(
            f"[ImageVisualiser] Showing RGB interpretation from bands -> R:{bands[0]}, G:{bands[1]}, B:{bands[2]}")

        plot_data = image_data.img_data[:, :, bands]

        plot_data /= (plot_data.max(axis=(0, 1)) + 1e-10)

        plt.imshow(plot_data)

    elif len(bands) == 1:
        logger.info(f"[ImageVisualiser] Showing band -> {bands[0]}")

        plot_data = image_data.img_data[:, :, bands]

        plot_data[:, :, 0] *= (1.0 / plot_data[:, :, 0].max())

        plt.imshow(plot_data)

    elif len(bands) == 0:
        logger.info("[ImageVisualiser] Showing image as a brightness plot")

        non_empty_bands = image_data.img_data.max(axis=(0, 1)) > 1e-2
        bands = [i for i, x in enumerate(non_empty_bands) if x]
        plot_data = image_data.img_data[:, :, (bands)]
        plot_data = plot_data.sum(axis=2)
        plot_data /= len(bands)
        plot_data *= (1.0 / plot_data.max())

        plt.imshow(plot_data, cmap='gray', vmin=0.0, vmax=1.0)

    else:
        logger.info(f"[ImageVisualiser] Wrong band choice was provided. Expected [] or len(bands) == 3, got {bands}")
        raise IncompatibleBandChoice


def plot_area_spectrum(image_data: ImageData, coordinates: AreaLocation) -> None:
    """ Plot spectrum of pixels 

    :param image_data: ImageData instance
    :param coordinates: AreaLocation instance
    :raises NoBandCenters: if provided ImageData contains no band_centers
    """

    ulx, uly, lrx, lry = coordinates.as_tuple()

    logger.info(f"[ImageVisualiser] Plotting mean spectrum of area x {ulx}:{lrx}, y {lry}:{uly}")

    if not image_data.band_centers:
        raise NoBandCenters("[ImageVisualiser] Cannot plot spectrum - band_centers are missing.")

    area_data = ImageData.mean_spectrum_area(image_data.img_data, coordinates.as_tuple())
    plt.plot(image_data.band_centers, area_data, label="Spectral response")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Spectral reflectance [-]")
