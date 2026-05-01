'''
Multispectral Camera Model - Image Visualiser
=============================================

Dataclasses and their methods for visualising image data
'''

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from matplotlib.pylab import isin

from ms_camera_model import AreaLocation, ImageData
from ms_camera_model.errors import IncompatibleBandChoice

logger = logging.getLogger(__name__)


class ImageVisualiser:
    """ Image Visualiser class

    :param image_data: ImageData class
    
    :method imshow: view the img_data as an RGB interpretation of selected bands or as a brightness plot
    :method plot_area_spectrum: visualize the mean spectrum of an area
    """

    @staticmethod
    def imshow(image_data: ImageData, bands: list[int] | None = None) -> None:
        """ View image as an RGB interpretation of selected bands or as a brightness plot

        :param bands: list of band numbers, if the list is not provided, brightness plot will be used instead
        :raises IncompatibleBandChoice: if 'bands' is not [], len(bands) != 3 or the number is out-of-bounds
        """

        if not bands:
            bands = []

        if not isinstance(bands, list):
            raise TypeError(f"Expected list[int] or None, got {type(bands)}")

        if len(bands) == 3:
            logger.info(
                f"[ImageData] Showing RGB interpretation from bands -> R:{bands[0]}, G:{bands[1]}, B:{bands[2]}")

            plot_data = image_data.img_data[:, :, bands]

            plot_data /= (plot_data.max(axis=(0, 1)) + 1e-10)

            plt.imshow(plot_data)

        elif len(bands) == 1:
            logger.info(f"[ImageData] Showing band -> {bands[0]}")

            plot_data = image_data.img_data[:, :, bands]

            plot_data[:, :, 0] *= (1.0 / plot_data[:, :, 0].max())

            plt.imshow(plot_data)

        elif len(bands) == 0:
            logger.info("[ImageData] Showing image as a brightness plot")

            non_empty_bands = image_data.img_data.max(axis=(0, 1)) > 1e-2
            bands = [i for i, x in enumerate(non_empty_bands) if x]
            plot_data = image_data.img_data[:, :, (bands)]
            plot_data = plot_data.sum(axis=2)
            plot_data /= len(bands)
            plot_data *= (1.0 / plot_data.max())

            plt.imshow(plot_data, cmap='gray', vmin=0.0, vmax=1.0)

        else:
            logger.info(f"[ImageData] Wrong band choice was provided. Expected [] or len(bands) == 3, got {bands}")
            raise IncompatibleBandChoice

    @staticmethod
    def plot_area_spectrum(image_data: ImageData, coordinates: AreaLocation) -> None:
        """ Plot spectrum of pixels 

        :param coordinates: list[ulx, uly, lrx, lry] - ulx means upper left x, lry means lower right y, etc.
        """

        ulx, uly, lrx, lry = coordinates.as_tuple()

        logger.info(f"[ImageData] Plotting mean spectrum of area x {ulx}:{lrx}, y {lry}:{uly}")

        if image_data.band_centers is None:
            logger.error("[ImageData] Cannot plot spectrum - band_centers are missing.")
            return

        area_data = image_data.mean_spectrum_area(image_data.img_data, coordinates.as_tuple())
        plt.plot(image_data.band_centers, area_data, label="Spectral response")
        plt.xlabel("Band")
        plt.ylabel("Reflectance")
