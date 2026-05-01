'''
Multispectral Camera Model - Image Data
=======================================

Dataclasses and their methods for image data
'''

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2 as cv
import micasense.metadata as metadata
import micasense.utils as msutils
import numpy as np
import pandas as pd
import spectral
from skimage import img_as_float

from .errors import (
    AreaOutsideOfBounds,
    ImageDataIncompatible,
    ImageImportFailed,
    NoDarkFrame,
    NoImageData,
    NoProvidedArea,
    NoProvidedFilepaths,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageData(ABC):
    """ Image Data class 

    :param img_data: np.ndarray of image data with shape (rows, cols, bands)
    :param band_centers: list of floats storing the center of each band in the img_data array
    :param nbands: number of bands in the img_data array

    :method __add__: add two ImageData classes
    :method mean_spectrum_area: calculate mean spectrum for a select area
    """

    img_data: np.ndarray
    band_centers: list[float]
    nbands: int

    @abstractmethod
    def _create_new_instance(self, new_img_data: np.ndarray):
        """ Create new instance of subclass

        :param new_img_data: """
        pass

    def __add__(self, other: ImageData) -> ImageData:
        """ Addition of two ImageData classes 
        
        :param self:    self
        :param other:   other ImageData class
        :raises ImageDataIncompatible: when the provided class instances do not have matching band_centers and number of bands
        """

        logger.info("[ImageData] Performing ImageData addition...")

        if not self.nbands == other.nbands or not np.allclose(self.band_centers, other.band_centers, atol=1e-3):
            raise ImageDataIncompatible("ImageData objects used for addition are not compatible with each other")

        return self._create_new_instance(self.img_data + other.img_data)

    @staticmethod
    def mean_spectrum_area(img: np.ndarray, corner_coords: tuple[int, int, int, int]) -> np.ndarray:
        """ Calculate mean over spectral bands for select area

        :param img: image data
        :param corner_coords: coordinates of corners of the area in format [ulx, uly, lrx, lry]
        """

        img_3d = np.atleast_3d(img)

        ulx, uly, lrx, lry = corner_coords

        shape = img.shape

        if lrx > shape[1] or lry > shape[0]:
            raise AreaOutsideOfBounds(f"Area bounds exceeded. Coordinates x:{lrx}, y:{lry} outside of {shape}")

        logger.info(f"[ImageData] Calculating mean for area x: {ulx}-{lrx}, y: {uly}-{lry}...")

        pixel_spectrum = img_3d[uly:lry, ulx:lrx, :]
        mean_spectrum = pixel_spectrum.mean(axis=(0, 1))

        logger.info("[ImageData] Mean calculation for selected area completed")

        return mean_spectrum


@dataclass(frozen=True)
class AreaLocation:
    """ Area location value object with checks """

    ulx: int
    uly: int
    lrx: int
    lry: int

    def __post_init__(self) -> None:
        """ Post init for checking coordinates """

        if self.ulx < 0 or self.lry < 0:
            raise ValueError("Area coordinates cannot be negative.")
        if self.ulx >= self.lrx:
            raise ValueError(f"Upper-left X ({self.ulx}) must be <= Lower-right X ({self.lrx})")
        if self.uly >= self.lry:
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

    def _create_new_instance(self, new_img_data: np.ndarray) -> ModeledMultispectralImageData:
        return ModeledMultispectralImageData(new_img_data, self.band_centers, self.nbands, self.band_names)

    def perform_radiometric_calibration(self, panel_calibration: dict[str, float],
                                        panel_locations: list[AreaLocation]) -> ModeledMultispectralImageData:
        """ Perform radiometric calibration on the modeled multispectral image data

        :param panel_calibration: panel_calibration data of used MicaSense CRP
        :param panel_location: list of AreaLocation objects
        """

        calibrated_img_data = np.zeros_like(self.img_data, dtype=np.float32)

        for i_band in range(self.img_data.shape[2]):
            logger.info(f"Performing radiometric calibration for modeled band {i_band}")

            mean_radiance = ImageData.mean_spectrum_area(self.img_data[:, :, i_band],
                                                         panel_locations[i_band].as_tuple())
            panel_reflectance = panel_calibration[self.band_names[i_band]]
            radiance_to_reflectance = panel_reflectance / mean_radiance

            calibrated_img_data[:, :, i_band] = self.img_data[:, :, i_band] * radiance_to_reflectance

        return self._create_new_instance(calibrated_img_data)


@dataclass
class MultispectralImageData(ImageData):
    """ Multispectral Image Data - loaded from real MS camera 

    :method import_altum_pt_ms_imgs: method for import of images from MicaSense Altum-PT camera
    :method import_ms_imgs: method for import of generic multispectral images
    :method check_filepaths: method which checks that filepaths are a non-empty list and are strings
    """

    def _create_new_instance(self, new_img_data: np.ndarray) -> MultispectralImageData:
        return MultispectralImageData(new_img_data, self.band_centers, self.nbands)

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
            raise ImageDataIncompatible(
                "Number of image filepaths doesn't match number of calibration panel coordinates")

        if not panel_locations:
            raise NoProvidedArea("No provided coordinates for calibration panel")

        MultispectralImageData.check_filepaths(filepaths)

        loaded_images = []
        band_centers = []

        for i_img, image_filepath in enumerate(filepaths):
            logger.info(f"[ImageData] Importing MicaSense multispectral image {image_filepath}...")

            try:
                img_raw = cv.imread(image_filepath, cv.IMREAD_UNCHANGED)

            except Exception as e:
                raise ImageImportFailed(f"Error {e} occured while reading image from file {image_filepath}.") from e

            if img_raw is None:
                raise NoImageData(f"Image read from {image_filepath} resulted in empty array.")

            meta = metadata.Metadata(image_filepath)

            radiance_img, *_ = msutils.raw_image_to_radiance(meta, img_raw)

            coordinates = panel_locations[i_img].as_tuple()
            mean_radiance = ImageData.mean_spectrum_area(radiance_img, coordinates)

            band_name = meta.get_item('XMP:BandName')
            band_centers.append(meta.get_item('XMP:CentralWavelength'))
            panel_reflectance = panel_calibration[band_name]
            radiance_to_reflectance = panel_reflectance / mean_radiance

            reflectance_img = radiance_img * radiance_to_reflectance

            loaded_images.append(img_as_float(reflectance_img).astype(np.float32))

        if not loaded_images:
            raise NoImageData("No image data was loaded from provided filepaths")

        logger.info("[ImageData] Import of MicaSense multispectral images completed")

        img_data = np.stack(loaded_images, axis=-1)
        nbands = len(loaded_images)

        return cls(img_data, band_centers, nbands)

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
                raise ImageImportFailed(f"Error {e} occured while reading image from file {image_filepath}.") from e

            if img is None:
                raise NoImageData(f"Image read from {image_filepath} resulted in empty array.")

            loaded_images.append(img_as_float(img).astype(np.float32))

        if not loaded_images:
            raise NoImageData("No image data was loaded from provided filepaths")

        img_data = np.stack(loaded_images, axis=-1)
        nbands = len(loaded_images)

        logger.info("[ImageData] Import of multispectral images completed")

        return cls(img_data, band_centers=[], nbands=nbands)

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

    def _create_new_instance(self, new_img_data: np.ndarray) -> HyperspectralImageData:
        return HyperspectralImageData(new_img_data, self.band_centers, self.nbands)

    @classmethod
    def import_calibrated_hs_img(cls, img_filepath: str, panel_data_filepath: str,
                                 panel_location: AreaLocation) -> HyperspectralImageData:
        """ Import hyperspectral cube as ImageData class instance 

        :param img_filepath: path to the hyperspectral image file
        :param panel_data_filepath: path to the csv panel albedo file
        :param panel_location: AreaLocation object
        :raises NoImageData: when spectral fails to load the image
        """

        if not img_filepath:
            raise NoProvidedFilepaths

        logger.info(f"[ImageData] Beginning import of hyperspectral file {img_filepath}...")

        try:
            img = spectral.open_image(img_filepath)

        except Exception as e:
            logger.error(f"[ImageData] Loading hyperspectral data from file {img_filepath} ended with error {e}")
            raise NoImageData from e

        img_data = img.load().astype(np.float32)
        metadata = img.metadata

        img_data_calibrated, valid_band_centers = HyperspectralImageData.perform_radiometric_calibration(
            panel_data_filepath, metadata, img_data, panel_location)

        nbands = img_data_calibrated.shape[2]
        valid_band_centers = list(map(float, valid_band_centers))

        logger.info("[ImageData] Hyperspectral image import completed")

        return cls(img_data_calibrated, valid_band_centers, nbands)

    @classmethod
    def import_hs_img(cls, img_filepath: str) -> HyperspectralImageData:
        """ Import hyperspectral cube as ImageData class instance 

        :param img_filepath: path to the hyperspectral image file
        :param panel_data_filepath: path to the csv panel albedo file
        :param panel_location: panel_location information format [ulx, uly, lrx, lry]
        :raises NoImageData: when spectral fails to load the image
        """

        if not img_filepath:
            raise NoProvidedFilepaths

        logger.info(f"[ImageData] Beginning import of hyperspectral file {img_filepath}...")

        try:
            img = spectral.open_image(img_filepath)

        except Exception as e:
            logger.error(f"[ImageData] Loading hyperspectral data from file {img_filepath} ended with error {e}")
            raise NoImageData from e

        img_data = img.load().astype(np.float32)
        band_centers = list(map(float, img.metadata['wavelength']))
        nbands = img.nbands

        logger.info("[ImageData] Hyperspectral image import completed")

        return cls(img_data, band_centers, nbands)

    @staticmethod
    def perform_radiometric_calibration(filepath: str,
                                        metadata: dict[str, str],
                                        img_data: np.ndarray,
                                        panel_location: AreaLocation,
                                        snr_multiplier: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """ Perform radiometric calibration based on calibration plate with known albedo

        :param filepath: path to the wavelength-albedo csv file
        :param metadata: dict with hyperspectral data metadata
        :param img_data: np.ndarray of image data with shape (rows, cols, bands)
        :param panel_location: panel_location information format [ulx, uly, lrx, lry]
        :param snr_multiplier: multiplier for snr check
        :return: calibrated numpy img_data array
        """

        if not filepath:
            raise NoProvidedFilepaths

        panel_calibration = pd.read_csv(filepath).to_numpy()
        panel_wavelengths = panel_calibration[:, 0]
        panel_reflectance = panel_calibration[:, 1]
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
        snr_threshold = snr_multiplier * dark_noise_sigma

        valid_data = img_data[:dark_frame_start, :, band_data_mask] - dark_frame_mean

        valid_wavelengths = hs_data_wavelengths[band_data_mask]

        white_reference = ImageData.mean_spectrum_area(valid_data, panel_location.as_tuple())

        valid_bands_mask = white_reference > snr_threshold
        dropped_bands_count = np.sum(~valid_bands_mask)
        if dropped_bands_count > 0:
            logger.warning(
                f"[RadiometricCalibration] Dropped {dropped_bands_count} bands due to failing the {snr_multiplier}-sigma SNR check."
            )

        interpolated_calibration_data = np.interp(valid_wavelengths, panel_wavelengths, panel_reflectance)

        calibration_factor = np.zeros_like(interpolated_calibration_data, dtype=np.float32)
        np.divide(interpolated_calibration_data, white_reference, out=calibration_factor, where=valid_bands_mask)

        calibrated_data = valid_data * calibration_factor

        calibrated_data = calibrated_data[:, :, valid_bands_mask]
        valid_wavelengths = valid_wavelengths[valid_bands_mask]

        return calibrated_data, valid_wavelengths
