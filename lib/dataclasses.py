'''
=======================================================================================================================
Name:           Dataclasses for Multispectral camera model
Description:    Dataclasses used for modeling multispectral camera modeling
Author:         Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
from math import ceil
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from lib.errors import ImgDataIncompatible


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
            raise ImgDataIncompatible("ImageData objects used for addition are not compatible with each other")

        return ImageData(self.img_data + other.img_data, self.band_centres,
                         self.num_of_bands)

    def imshow_rgb(self) -> None:
        """ View image as an RGB - get three bands from usable part of spectrum (start, middle, end) """

        non_empty_bands = self.img_data[(ceil(self.img_data.shape[0] / 2)),
                                        (ceil(self.img_data.shape[1] /
                                              2)), :] > 1e-2
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

        area_data = self.get_mean_spectrum_of_area(top_right_coordinate,
                                                   pixels_per_dimension)
        plt.plot(self.band_centres, area_data, label="Spectral response")
        plt.xlabel("Band")
        plt.ylabel("Reflectance")
        plt.show()

    # Got this from Gemini
    def vector_normalize(self) -> None:
        """ Normalize img data """
        # Reshape the image data to a 2D array (pixels x bands)
        pixels = self.img_data.reshape(-1, self.img_data.shape[2])

        # Calculate the Euclidean norm (vector length) for each pixel's spectrum
        # The norm is the square root of the sum of the squares of all band values.
        norms = np.linalg.norm(pixels, axis=1)

        # Avoid division by zero for pixels with a norm of 0
        norms[norms == 0] = 1e-10

        # Divide each pixel's spectrum by its norm
        normalized_pixels = pixels / norms[:, np.newaxis]

        # Reshape the data back to the original image shape
        self.img_data = normalized_pixels.reshape(self.img_data.shape)

    def get_mean_spectrum_of_area(
            self,
            corner_coords: list[int] = [946, 349],
            pixels_per_dimension: int = 5) -> list[np.float64] | None:
        """ Get mean spectrum of a selected square area

        :param img: image data
        :param square_corner_coordinates: coordinates of an upper left corner of the square area
        :param pixels_per_dimension: number of pixels per dimension
        """

        pixel_spectrum = self.img_data[corner_coords[0]:corner_coords[0] +
                                       pixels_per_dimension,
                                       corner_coords[1]:corner_coords[1] +
                                       pixels_per_dimension, :]
        mean_spectrum = pixel_spectrum.mean(axis=(0, 1))

        return list(mean_spectrum)


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
