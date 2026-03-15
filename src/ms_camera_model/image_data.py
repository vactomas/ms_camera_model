'''
Multispectral Camera Model - Image Data
=======================================

Dataclasses and their methods for image data
'''

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import micasense.metadata as metadata
import micasense.utils as msutils
import numpy as np
import pandas as pd
import spectral
from skimage import img_as_float

from .errors import (
    ImageDataIncompatible,
    IncompatibleBandChoice,
    NoDarkFrame,
    NoImageData,
    NoProvidedArea,
    NoProvidedFilepaths,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """ Image Data class 

    :param img_data: np.ndarray of image data with shape (rows, cols, bands)
    :param band_centers: list of floats storing the center of each band in the img_data array
    :param nbands: number of bands in the img_data array

    :method __add__: add two ImageData classes
    :method imshow: view the img_data as an RGB interpretation of selected bands or as a brightness plot
    :method vector_normalize: use vector normalization on the img_data
    :method mean_spectrum_area: calculate mean spectrum for a select area
    :method plot_area_spectrum: visualize the mean spectrum of an area
    :method register_images: register img_data of another ImageData class instance against this instance
    """

    img_data: np.ndarray
    band_centers: list[float]
    nbands: int

    def __add__(self, other: ImageData) -> ImageData:
        """ Addition of two ImageData classes 
        
        :param self:    self
        :param other:   other ImageData class
        :raises ImageDataIncompatible: when the provided class instances do not have matching band_centers and number of bands
        """

        logger.info("[ImageData] Performing ImageData addition...")

        if not self.nbands == other.nbands or not self.band_centers == other.band_centers:
            raise ImageDataIncompatible("ImageData objects used for addition are not compatible with each other")

        return ImageData(self.img_data + other.img_data, self.band_centers, self.nbands)

    def imshow(self, bands: list[int] | None = None) -> None:
        """ View image as an RGB interpretation of selected bands or as a brightness plot

        :param bands: list of band numbers, if the list is not provided, brightness plot will be used instead
        :raises IncompatibleBandChoice: if 'bands' is not [], len(bands) != 3 or the number is out-of-bounds
        """

        if not bands:
            bands = []

        if len(bands) == 3:
            logger.info(
                f"[ImageData] Showing RGB interpretation from bands -> R:{bands[0]}, G:{bands[1]}, B:{bands[2]}")

            plot_band_list = [bands[0], bands[1], bands[2]]
            plot_data = self.img_data[:, :, (plot_band_list)]

            # Rescale values to range 0.0 - 1.0 for each band individually
            # for i in range(len(bands)):
            #     plot_data[:, :, i] *= (1.0 / plot_data[:, :, i].max())

            plot_data /= (plot_data.max(axis=(0, 1)) + 1e-10)

            plt.imshow(plot_data)

        elif len(bands) == 1:
            logger.info(f"[ImageData] Showing band -> {bands[0]}")

            plot_band_list = [bands[0]]
            plot_data = self.img_data[:, :, (plot_band_list)]

            plot_data[:, :, 0] *= (1.0 / plot_data[:, :, 0].max())

            plt.imshow(plot_data)

        elif bands == []:
            logger.info("[ImageData] Showing image as a brightness plot")

            non_empty_bands = self.img_data.max(axis=(0, 1)) > 1e-2
            bands = [i for i, x in enumerate(non_empty_bands) if x]
            plot_data = self.img_data[:, :, (bands)]
            plot_data = plot_data.sum(axis=2)
            plot_data /= len(bands)
            plot_data *= (1.0 / plot_data.max())

            plt.imshow(plot_data, cmap='gray', vmin=0.0, vmax=1.0)

        else:
            logger.info(f"[ImageData] Wrong band choice was provided. Expected [] or len(bands) == 3, got {bands}")
            raise IncompatibleBandChoice

    def plot_area_spectrum(self, coordinates: list[int]) -> None:
        """ Plot spectrum of pixels 

        :param coordinates: list[ulx, uly, lrx, lry] - ulx means upper left x, lry means lower right y, etc.
        """

        logger.info(
            f"[ImageData] Plotting mean spectrum of area x {coordinates[0]}:{coordinates[2]}, y {coordinates[1]}:{coordinates[3]}"
        )

        if self.band_centers is None:
            logger.error("[ImageData] Cannot plot spectrum - band_centers are missing.")
            return

        area_data = self.mean_spectrum_area(self.img_data, coordinates)
        plt.plot(self.band_centers, area_data, label="Spectral response")
        plt.xlabel("Band")
        plt.ylabel("Reflectance")

    def vector_normalize(self) -> None:
        """ Normalize img data """

        logger.info("[ImageData] Performing vector normalization on the img_data...")

        pixels = self.img_data.reshape(-1, self.img_data.shape[2])
        norms = np.linalg.norm(pixels, axis=1)

        # Avoid division by zero for pixels with a norm of 0
        norms[norms == 0] = 1e-10

        # Divide each pixel's spectrum by its norm
        normalized_pixels = pixels / norms[:, np.newaxis]
        self.img_data = normalized_pixels.reshape(self.img_data.shape)

        logger.info("[ImageData] Vector normalization completed")

    @staticmethod
    def mean_spectrum_area(img: np.ndarray, corner_coords: tuple[int, int, int, int]) -> np.ndarray:
        """ Calculate mean over spectral bands for select area

        :param img: image data
        :param corner_coords: coordinates of corners of the area in format [ulx, uly, lrx, lry]
        """

        img_3d = np.atleast_3d(img)

        ulx, uly, lrx, lry = corner_coords

        logger.info(f"[ImageData] Calculating mean for area x: {ulx}-{lrx}, y: {uly}-{lry}...")

        pixel_spectrum = img_3d[uly:lry, ulx:lrx, :]
        mean_spectrum = pixel_spectrum.mean(axis=(0, 1))

        logger.info("[ImageData] Mean calculation for selected area completed")

        return mean_spectrum

    def normalize_img_data(self) -> None:
        """ Normalize image data """

        max_value = np.max(self.img_data)

        self.img_data = self.img_data / max_value


@dataclass(frozen=True)
class AreaLocation:
    """ Area location value object with checks """

    ulx: int
    uly: int
    lrx: int
    lry: int

    def __post_init__(self) -> None:
        """ Post init for checking coordinates """

        if self.ulx > self.lrx:
            raise ValueError(f"Upper-left X ({self.ulx}) must be <= Lower-right X ({self.lrx})")
        if self.uly > self.lry:
            raise ValueError(f"Upper-left Y ({self.uly}) must be <= Lower-right Y ({self.lry})")

    def as_tuple(self) -> tuple[int, int, int, int]:
        """ Return in openCV compatible format """

        return (self.ulx, self.uly, self.lrx, self.lry)


@dataclass
class ModeledMultispectralImageData(ImageData):
    """ Modeled Multispectral Image Data - modeled from HS data

    :method perform_radiometric_calibration: method for performing radiometric calibration against known panel
    """

    band_names: list[str]

    def perform_radiometric_calibration(self, panel_calibration: dict[str, float],
                                        panel_locations: list[AreaLocation]) -> ModeledMultispectralImageData:
        """ Perform radiometric calibration on the modeled multispectral image data

        :param panel_calibration: panel_calibration data of used MicaSense CRP
        :param panel_location: list of AreaLocation objects
        """

        calibrated_img_data = np.zeros_like(self.img_data)

        for i_band in range(self.img_data.shape[2]):
            logger.info(f"Performin radiometric calibration for modeled band {i_band}")

            mean_radiance = ImageData.mean_spectrum_area(self.img_data[:, :, i_band],
                                                         panel_locations[i_band].as_tuple())
            panel_reflectance = panel_calibration[self.band_names[i_band]]
            radiance_to_reflectance = panel_reflectance / mean_radiance

            calibrated_img_data[:, :, i_band] = self.img_data[:, :, i_band] * radiance_to_reflectance

        return ModeledMultispectralImageData(calibrated_img_data, self.band_centers, self.nbands, self.band_names)


@dataclass
class MultispectralImageData(ImageData):
    """ Multispectral Image Data - loaded from real MS camera 

    :method import_altum_pt_ms_imgs: method for import of images from MicaSense Altum-PT camera
    :method import_ms_imgs: method for import of generic multispectral images
    :method check_filepaths: method which checks that filepaths are a non-empty list and are strings
    """

    @classmethod
    def import_altum_pt_ms_imgs(cls, filepaths: list[str], panel_calibration: dict[str, float],
                                panel_locations: list[AreaLocation]) -> MultispectralImageData:
        """ Import and pre-process Altum PT images 

        :param filepaths: list of filepaths to the multispectral images, their order determines order in the final array
        :param panel_calibration: panel_calibration data of used MicaSense CRP
        :param panel_location: list of AreaLocation objects
        :raises NoProvidedArea: when the provided area of the CRP is None or empty
        :raises Exception: when the length of filepaths and panel_locations doesn't match
        :raises TypeError: when provided paths aren't in list
        :raises TypeError: when provided paths aren't strings
        :raises NoProvidedFilepaths: when there are no provided filepaths
        """

        logger.info("[ImageData] Beginning import of multispectral images...")

        if not len(panel_locations) == len(filepaths):
            raise Exception("Number of image filepaths doesn't match number of calibration panel coordinates")

        if not panel_locations or panel_locations is None:
            raise NoProvidedArea("No provided coordinates for calibration panel")

        MultispectralImageData.check_filepaths(filepaths)

        loaded_images = []
        band_centers = []

        for i_img, image_filepath in enumerate(filepaths):
            logger.info(f"[ImageData] Importing MicaSense multispectral image {image_filepath}...")

            try:
                img_raw = cv.imread(image_filepath, cv.IMREAD_UNCHANGED)

            except Exception as e:
                logger.warning(
                    f"[ImageData] Error {e} occured while reading image from file {image_filepath}. Skipping...")
                continue

            meta = metadata.Metadata(image_filepath)

            radiance_img, *_ = msutils.raw_image_to_radiance(meta, img_raw)

            coordinates = panel_locations[i_img].as_tuple()
            mean_radiance = ImageData.mean_spectrum_area(radiance_img, coordinates)

            band_name = meta.get_item('XMP:BandName')
            band_centers.append(meta.get_item('XMP:CentralWavelength'))
            panel_reflectance = panel_calibration[band_name]
            radiance_to_reflectance = panel_reflectance / mean_radiance

            reflectance_img = radiance_img * radiance_to_reflectance

            loaded_images.append(img_as_float(reflectance_img))

        if not loaded_images:
            raise NoImageData("No image data was loaded from provided filepaths")

        logger.info("[ImageData] Import of MicaSense multispectral images completed")

        img_data = np.stack(loaded_images, axis=-1)
        nbands = len(loaded_images)

        return MultispectralImageData(img_data, band_centers, nbands)

    @classmethod
    def import_ms_imgs(cls, filepaths: list[str]) -> MultispectralImageData:
        """ Import multispectral images as a np.ndarray 

        :param filepaths: list of filepaths to the multispectral images, their order determines order in the final array
        :raises TypeError: when provided paths aren't in list
        :raises TypeError: when provided paths aren't strings
        :raises NoProvidedFilepaths: when there are no provided filepaths
        """

        logger.info("[ImageData] Beginning import of multispectral images...")

        MultispectralImageData.check_filepaths(filepaths)

        loaded_images = []

        for image_filepath in filepaths:
            logger.info(f"[ImageData] Importing multispectral image {image_filepath}...")

            try:
                img = cv.imread(image_filepath, cv.IMREAD_UNCHANGED)

            except Exception as e:
                logger.warning(
                    f"[ImageData] Error {e} occured while reading image from file {image_filepath}. Skipping...")
                continue

            loaded_images.append(img_as_float(img))

        if not loaded_images:
            raise NoImageData("No image data was loaded from provided filepaths")

        img_data = np.stack(loaded_images, axis=-1)
        nbands = len(loaded_images)

        logger.info("[ImageData] Import of multispectral images completed")

        return MultispectralImageData(img_data, nbands=nbands)

    @staticmethod
    def check_filepaths(filepaths: list[str]) -> None:
        """ Check that provided list of filepaths is usable

        :param filepaths: list of filepaths
        :raises TypeError: when provided paths aren't in list
        :raises TypeError: when provided paths aren't strings
        :raises NoProvidedFilepaths: when there are no provided filepaths
        """

        if not isinstance(filepaths, list):
            raise TypeError("Error. This function requires a list of paths.")

        if not all(isinstance(item, str) for item in filepaths):
            raise TypeError("Error. Some provided paths aren't strings.")

        if not filepaths:
            raise NoProvidedFilepaths


@dataclass
class HyperspectralImageData(ImageData):
    """ Hyperspectral Image Data - imported from hyperspectral data file 

    :method import_hs_img: Import hyperspectral cube into img_data, band_centers and nbands
    """

    @classmethod
    def import_calibrated_hs_img(cls, img_filepath: str, panel_data_filepath: str,
                                 panel_location: AreaLocation) -> HyperspectralImageData:
        """ Import hyperspectral cube as ImageData class instance 

        :param img_filepath: path to the hyperspectral image file
        :param panel_data_filepath: path to the csv panel albedo file
        :param panel_location: AreaLocation object
        :raises NoImageData: when spectral fails to load the image
        """

        logger.info(f"[ImageData] Beginning import of hyperspectral file {img_filepath}...")

        try:
            img = spectral.open_image(img_filepath)

        except Exception as e:
            logger.error(f"[ImageData] Loading hyperspectral data from file {img_filepath} ended with error {e}")
            raise NoImageData from e

        img_data = img.load()
        metadata = img.metadata

        img_data_calibrated, valid_band_centers = HyperspectralImageData.perform_radiometric_calibration(
            panel_data_filepath, metadata, img_data, panel_location.as_tuple())

        nbands = img_data_calibrated.shape[2]
        valid_band_centers = list(map(float, valid_band_centers))

        logger.info("[ImageData] Hyperspectral image import completed")

        return HyperspectralImageData(img_data_calibrated, valid_band_centers, nbands)

    @classmethod
    def import_hs_img(cls, img_filepath: str) -> HyperspectralImageData:
        """ Import hyperspectral cube as ImageData class instance 

        :param img_filepath: path to the hyperspectral image file
        :param panel_data_filepath: path to the csv panel albedo file
        :param panel_location: panel_location information format [ulx, uly, lrx, lry]
        :raises NoImageData: when spectral fails to load the image
        """

        logger.info(f"[ImageData] Beginning import of hyperspectral file {img_filepath}...")

        try:
            img = spectral.open_image(img_filepath)

        except Exception as e:
            logger.error(f"[ImageData] Loading hyperspectral data from file {img_filepath} ended with error {e}")
            raise NoImageData from e

        img_data = img.load()
        band_centers = img.metadata['wavelength']
        nbands = img.nbands

        logger.info("[ImageData] Hyperspectral image import completed")

        return HyperspectralImageData(img_data, band_centers, nbands)

    @staticmethod
    def perform_radiometric_calibration(filepath: str, metadata: dict[str, str], img_data: np.ndarray,
                                        panel_location: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        """ Perform radiometric calibration based on calibration plate with known albedo

        :param filepath: path to the wavelength-albedo csv file
        :param metadata: dict with hyperspectral data metadata
        :param img_data: np.ndarray of image data with shape (rows, cols, bands)
        :param panel_location: panel_location information format [ulx, uly, lrx, lry]
        :return: calibrated numpy img_data array
        """

        panel_calibration = pd.read_csv(filepath)
        panel_wavelengths = panel_calibration.iloc[:, 0].values
        panel_reflectance = panel_calibration.iloc[:, 1].values
        panel_validity_start, panel_validity_end = [panel_wavelengths.min(), panel_wavelengths.max()]

        if 'autodarkstartline' not in metadata:
            raise NoDarkFrame

        dark_frame_start = int(metadata['autodarkstartline'])
        hs_data_wavelengths = np.array(list(map(float, metadata['wavelength'])))
        band_data_mask = (hs_data_wavelengths >= panel_validity_start) & (hs_data_wavelengths <= panel_validity_end)
        band_data_mask[0] = False

        dark_frame = img_data[dark_frame_start:, :, band_data_mask]
        dark_frame_mean = np.mean(dark_frame, axis=0)

        dark_noise_sigma = np.std(dark_frame, axis=(0, 1))
        snr_threshold = 5.0 * dark_noise_sigma

        valid_data = img_data[:dark_frame_start, :, band_data_mask] - dark_frame_mean

        valid_wavelengths = hs_data_wavelengths[band_data_mask]

        white_reference = ImageData.mean_spectrum_area(valid_data, panel_location)

        valid_bands_mask = white_reference > snr_threshold
        dropped_bands_count = np.sum(~valid_bands_mask)
        if dropped_bands_count > 0:
            logger.warning(
                f"[RadiometricCalibration] Dropped {dropped_bands_count} bands due to failing the 5-sigma SNR check.")

        interpolated_calibration_data = np.interp(valid_wavelengths, panel_wavelengths, panel_reflectance)

        calibration_factor = np.zeros_like(interpolated_calibration_data)
        np.divide(interpolated_calibration_data, white_reference, out=calibration_factor, where=valid_bands_mask)

        calibrated_data = valid_data * calibration_factor

        calibrated_data = calibrated_data[:, :, valid_bands_mask]
        valid_wavelengths = valid_wavelengths[valid_bands_mask]

        return calibrated_data, valid_wavelengths
