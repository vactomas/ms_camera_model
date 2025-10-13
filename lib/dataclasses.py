'''
=======================================================================================================================
- Name:         Dataclasses for Multispectral camera model
- Description:  Dataclasses used for modeling multispectral camera modeling
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
from math import ceil
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from lib.errors import ImgDataIncompatible
import cv2 as cv
import logging

from skimage import img_as_float

logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M",)


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

    def imshow_rgb(self, bands: list[int]) -> None:
        """ View image as an RGB interpretation of selected bands """

        logging.debug(f"Showing RGB interpretation from bands -> R:{bands[0]}, G:{bands[1]}, B:{bands[2]}")

        plot_band_list = [bands[0], bands[1], bands[2]]
        plot_data = self.img_data[:, :, (plot_band_list)]
        # plot_data *= (1.0 / plot_data.max())

        # Rescale values to range 0.0 - 1.0 for each band individually
        for i in range(len(bands)):
            plot_data[:, :, i] *= (1.0 / plot_data[:, :, i].max())

        plt.imshow(plot_data)
        plt.show()

    def imshow(self) -> None:
        """ View image as a brightness plot """

        logging.debug("Showing image as a brightness plot")

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

        logging.debug("Performing vector normalization on the image data.")

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

        logging.debug(f"Calculating mean spectrum for area x: {corner_coords[0]}-{corner_coords[0] + pixels_per_dimension}, y: {corner_coords[1]}-{corner_coords[1] + pixels_per_dimension}")

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

        logging.debug(f"Importing filter specification from file {filename}")

        excel_data = pd.read_excel(filename)
        self.filter_transmission = np.array(excel_data)


@dataclass
class SensorSpecs:
    """ Sensor specification """
    sensor_qe_curve: np.ndarray
    name: str = "Generic"
    supplier: str = "Generic"
    sensor_type: str = "CMOS"


@dataclass
class FilterSensorUnit:
    """ Combination of filter and sensor """
    filter_spec: FilterSpecs
    sensor_spec: SensorSpecs
    combined_attenuation: np.ndarray

    def load_filter_sensor(self, filename_filter: str, filename_sensor: str) -> None:
        """ Load filter transmission and sensor spectral sensitivity from a file

        :param filename_filter: Filename or path to the filter xlsx file
        :param filename_sensor: Filename or path to the sensor xlsx file
        """

        logging.debug(f"Loading specifications\nFilter data: {filename_filter}\nSensor data: {filename_sensor}")

        filter_data = pd.read_excel(filename_filter)
        sensor_data = pd.read_excel(filename_sensor)

        self.filter_spec.filter_transmission = np.array(filter_data)
        self.sensor_spec.sensor_qe_curve = np.array(sensor_data)

    def calculate_combined_attenuation(self) -> None:
        """ Calculate combined attenuation of filter and sensor based on their transmission and qe (quantum efficiency) curve """

        logging.debug("Calculating combined attenuation")

        self.combined_attentuation = self.filter_spec.filter_transmission[:, 1] * self.sensor_spec.sensor_qe_curve[:, 1]

class MultispectralCameraData:
    """ Data from a real multispectral camera """

    def __init__(self, band_files: list[str], band_centres: list[float], n_matches: int = 50) -> None:
        self.list_of_band_files: list[str] = band_files
        self.n_matches: int = n_matches
        self.combined_img: ImageData = ImageData(self.register_images(), band_centres, len(band_files))

    def register_images(self) -> np.ndarray:
        """ Register all images and return single array with data """
      
        if self.list_of_band_files is []:
            raise ValueError("No file paths were provided.")

        for band_num in range(len(self.list_of_band_files)):
            
            if band_num == 0:
                im_ref = cv.imread(self.list_of_band_files[0], cv.IMREAD_UNCHANGED)
                im_ref = img_as_float(im_ref)

                out_data = np.zeros((im_ref.shape[0], im_ref.shape[1], len(self.list_of_band_files)))
                out_data[:, :, 0] = im_ref

                continue

            out_data[:, :, band_num] = self.register_band(band_num)

        return out_data

    def register_band(self, band_num: int = 1) -> np.ndarray:
        """ Register band against reference """

        img_ref = cv.imread(self.list_of_band_files[0], cv.IMREAD_GRAYSCALE)
        img_src = cv.imread(self.list_of_band_files[band_num], cv.IMREAD_GRAYSCALE)

        finder = cv.ORB_create()
        kp_ref, des_ref = finder.detectAndCompute(img_ref, None)
        kp_src, des_src = finder.detectAndCompute(img_src, None)

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_ref, des_src)
        matches = sorted(matches, key = lambda x:x.distance)

        # Take only best 90% of matches
        matches = matches[:int(len(matches)*90)]

        point_ref = np.zeros((len(matches), 2))
        point_src = np.zeros((len(matches), 2))

        for i in range(len(matches)):
            point_ref[i, :] = kp_ref[matches[i].queryIdx].pt
            point_src[i, :] = kp_src[matches[i].trainIdx].pt

        logging.debug(f"Found {len(point_ref)} points")

        try:
            homography, _ = cv.findHomography(point_src, point_ref, cv.RANSAC)
            transformed_img = cv.warpPerspective(img_src, homography, (img_ref.shape[1], img_ref.shape[0]))

        except Exception as ex:
            logging.info(f"Not enough matched points found for band {band_num + 1}. Image registration unsuccessful. Returning array of zeros instead.")
            transformed_img = np.zeros((img_ref.shape[0], img_ref.shape[1]))

        return transformed_img


