'''
Multispectral Camera Model - Data Comparison
============================================

* **Description:** Classes and their methods used for comparing real MS data with modeled data
* **Author:** Tomas Vacek
'''
import logging

import numpy as np

from ms_camera_model.errors import (
    ImageDataIncompatible,
    InvalidProvidedArea,
    NoProvidedArea,
)
from ms_camera_model.image_data import (
    AreaLocation,
    ImageData,
    ModeledMultispectralImageData,
    MultispectralImageData,
)

logger = logging.getLogger(__name__)


class DataComparator:

    @staticmethod
    def compare_band_ratios(real_ms_image_data: MultispectralImageData,
                            modeled_ms_image_data: ModeledMultispectralImageData, real_ms_area_location: AreaLocation,
                            modeled_ms_area_location: AreaLocation) -> tuple[np.ndarray, np.ndarray]:
        """ Compare band ratios of real MS image data with modeled MS image data, area defined globally

        :param real_ms_image_data: MS image data from a real camera
        :param modeled_ms_image_data: modeled MS image data
        :param real_ms_area_location: AreaLocation object describing the area that will be compared
        :param modeled_ms_square_mean: AreaLocation object describing the area that will be compared
        :return: tuple(real_ms_ratios, modeled_ms_ratios)
        :raises ValueError: if sum of means of selected area is less than 1e-10
        :raises InvalidProvidedArea: 
        """

        logging.info("[DataComparator] Preparing comparison...")

        DataComparator._check_image_data_compatibility(real_ms_image_data, modeled_ms_image_data, real_ms_area_location,
                                                       modeled_ms_area_location)

        if isinstance(real_ms_area_location, list) or isinstance(modeled_ms_area_location, list):
            raise InvalidProvidedArea(
                f"Expected single AreaLocation object, got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
            )

        real_ms_square_mean = ImageData.mean_spectrum_area(real_ms_image_data.img_data,
                                                           real_ms_area_location.as_tuple())
        modeled_ms_square_mean = ImageData.mean_spectrum_area(modeled_ms_image_data.img_data,
                                                              modeled_ms_area_location.as_tuple())

        real_ms_ratios, modeled_ms_ratios = DataComparator._calculate_band_ratios(real_ms_square_mean,
                                                                                  modeled_ms_square_mean)

        return real_ms_ratios, modeled_ms_ratios

    @staticmethod
    def compare_band_ratios_per_band(real_ms_image_data: MultispectralImageData,
                                     modeled_ms_image_data: ModeledMultispectralImageData,
                                     real_ms_area_location: list[AreaLocation],
                                     modeled_ms_area_location: list[AreaLocation]) -> tuple[np.ndarray, np.ndarray]:
        """ Compare band ratios of real MS image data with modeled MS image data, area defined per band

        :param real_ms_image_data: MS image data from a real camera
        :param modeled_ms_image_data: modeled MS image data
        :param real_ms_area_location: list[AreaLocation] objects describing the area that will be compared
        :param modeled_ms_square_mean: list[AreaLocation] object describing the area that will be compared
        :return: tuple(real_ms_ratios, modeled_ms_ratios)
        :raises ValueError: if sum of means of selected area is less than 1e-10
        """

        DataComparator._check_image_data_compatibility(real_ms_image_data, modeled_ms_image_data, real_ms_area_location,
                                                       modeled_ms_area_location)

        if not isinstance(real_ms_area_location, list) or not isinstance(modeled_ms_area_location, list):
            raise InvalidProvidedArea(
                f"Expected list of AreaLocation objects for set_areas_globally = False, got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
            )

        if len(real_ms_area_location) != real_ms_image_data.nbands:
            raise InvalidProvidedArea(
                f"Provided area locations ({len(real_ms_area_location)}) does not match the number of bands ({real_ms_image_data.nbands})"
            )

        real_ms_square_mean = np.zeros(real_ms_image_data.nbands, dtype=np.float32)
        modeled_ms_square_mean = np.zeros(modeled_ms_image_data.nbands, dtype=np.float32)

        for band in range(real_ms_image_data.nbands):
            real_ms_square_mean[band] = ImageData.mean_spectrum_area(real_ms_image_data.img_data[:, :, band],
                                                                     real_ms_area_location[band].as_tuple())[0]
            modeled_ms_square_mean[band] = ImageData.mean_spectrum_area(modeled_ms_image_data.img_data[:, :, band],
                                                                        modeled_ms_area_location[band].as_tuple())[0]

        real_ms_ratios, modeled_ms_ratios = DataComparator._calculate_band_ratios(real_ms_square_mean,
                                                                                  modeled_ms_square_mean)

        return real_ms_ratios, modeled_ms_ratios

    @staticmethod
    def _check_image_data_compatibility(real_ms_image_data: MultispectralImageData,
                                        modeled_ms_image_data: ModeledMultispectralImageData,
                                        real_ms_area_location: AreaLocation | list[AreaLocation],
                                        modeled_ms_area_location: AreaLocation | list[AreaLocation]) -> None:

        if not real_ms_area_location or not modeled_ms_area_location:
            raise NoProvidedArea

        if real_ms_image_data.nbands != modeled_ms_image_data.nbands:
            raise ImageDataIncompatible(
                f"Provided image data has incompatible number of bands ({real_ms_image_data.nbands} vs {modeled_ms_image_data.nbands})"
            )

    @staticmethod
    def _calculate_band_ratios(real_ms_square_mean: np.ndarray,
                               modeled_ms_square_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Calculate band ratios 

        :param real_ms_square_mean: 1D array of means for defined area on real camera image data
        :param modeled_ms_square_mean: 1D array of means for defined area on modeled image data
        """
        sum_real_ms_mean = np.sum(real_ms_square_mean)
        sum_modeled_ms_mean = np.sum(modeled_ms_square_mean)

        if sum_real_ms_mean < 1e-10 or sum_modeled_ms_mean < 1e-10:
            raise ValueError("Denominator for next operation is 0 or close to 0")

        real_ms_ratios = real_ms_square_mean / sum_real_ms_mean
        modeled_ms_ratios = modeled_ms_square_mean / sum_modeled_ms_mean

        logging.info(
            f"[DataComparator] SQR_mean: {real_ms_square_mean}, SUM: {np.sum(real_ms_square_mean)}, ratios: {real_ms_ratios}"
        )
        logging.info(
            f"[DataComparator] SQR_mean: {modeled_ms_square_mean}, SUM: {np.sum(modeled_ms_square_mean)}, ratios: {modeled_ms_ratios}"
        )

        return real_ms_ratios, modeled_ms_ratios

    @staticmethod
    def calculate_spectral_angle_mapper(real_ms_ratios: np.ndarray, modeled_ms_ratios: np.ndarray) -> float:
        """ Calculate the spectral angle mapper (shape similarity) between real and modeled data
    
        :param real_ms_ratios: real MS band ratios
        :param modeled_ms_ratios: modeled MS band ratios
        :return: angle in rad describing the similarity independent of brightness
        :raises ValueError: if the calculation would cause division by 0
        """

        numerator = np.sum(real_ms_ratios * modeled_ms_ratios)
        norm_real = np.linalg.norm(real_ms_ratios)
        norm_modeled = np.linalg.norm(modeled_ms_ratios)
        denominator = norm_real * norm_modeled

        if denominator < 1e-10:
            logger.error("[DataComparator] Calculation leads to zero division")
            raise ValueError("Invalid denominator value for SAM calculation")

        val = np.clip(numerator / denominator, -1.0, 1.0)

        angle = np.arccos(val)

        return angle
