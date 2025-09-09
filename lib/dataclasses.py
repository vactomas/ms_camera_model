'''
=======================================================================================================================
Name:           Dataclasses for Multispectral camera model
description:    dataclasses used for modeling multispectral camera modeling
=======================================================================================================================
'''

from __future__ import annotations
from math import ceil
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from lib.spectral_scripts import get_mean_spectrum_of_area
from lib.errors import ImgDataIncompatibleError


@dataclass
class ImageData:
    """ Image Data class """
    img_data: np.ndarray  # ndarray with shape (rows, cols, bands)
    band_centres: list[float]
    num_of_bands: int

    def __add__(self, other: ImageData) -> ImageData:
        """ Addition of two ImageData classes 
        
        :param self:    self
        :param other:   other ImageData class
        """

        if not self.num_of_bands == other.num_of_bands and not self.band_centres == other.band_centres:
            raise ImgDataIncompatibleError

        return ImageData(self.img_data + other.img_data, self.band_centres,
                         self.num_of_bands)

    def imshow_rgb(self) -> None:
        """ View image as an RGB - get three bands from usable part of spectrum (start, middle, end) """

        non_empty_bands = self.img_data[(ceil(self.img_data.shape[0] / 2)),
                                        (ceil(self.img_data.shape[1] /2)), :] > 1e-2
        bands = [i for i, x in enumerate(non_empty_bands) if x]

        plot_band_list = [bands[0], bands[ceil(len(bands) / 2)], bands[-1]]
        plot_data = self.img_data[:, :, (plot_band_list)]
        plot_data *= (1.0 / plot_data.max())

        plt.imshow(plot_data)
        plt.show()

    def imshow(self) -> None:
        """ View image as a brightness plot """

        non_empty_bands = self.img_data[(ceil(self.img_data.shape[0] / 2)),
                                        (ceil(self.img_data.shape[1] /
                                              2)), :] > 1e-2
        bands = [i for i, x in enumerate(non_empty_bands) if x]
        plot_data = self.img_data[:, :, (bands)]
        plot_data = plot_data.sum(axis=2)
        plot_data /= len(bands)
        plot_data *= (1.0 / plot_data.max())

        plt.imshow(plot_data, cmap='gray', vmin=0.0, vmax=1.0)
        plt.show()

    def plot_area_spectrum(self,
                           top_right_coordinate: list[int],
                           pixels_per_dimension: int = 5) -> None:
        """ Plot spectrum of pixels """

        area_data = get_mean_spectrum_of_area(self.img_data,
                                              top_right_coordinate,
                                              pixels_per_dimension)
        plt.plot(self.band_centres, area_data, label="Spectral response")
        plt.xlabel("Band")
        plt.ylabel("Reflectance")
        plt.show()


@dataclass
class FilterSpecs:
    """ Filter specification """
    filter_transmission: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    band_center: int = 0
    band_width: int = 0

    def import_filter_specs(self, filename: str) -> None:
        """ Import filter specs from Excel data

        :param filename:    Filename or path to the filter xlsx file
        """
        excel_data = pd.read_excel(filename)
        self.filter_transmission = np.array(excel_data)
