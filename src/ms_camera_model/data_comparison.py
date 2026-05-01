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
from ms_camera_model.image_data import AreaLocation, ImageData

logger = logging.getLogger(__name__)


class DataComparator:

    def __init__(self, ms_img_data: ImageData, modeled_img_data: ImageData) -> None:

        self.ms_img_data: ImageData = ms_img_data
        self.modeled_img_data: ImageData = modeled_img_data

    def compare_band_ratios(self,
                            real_ms_area_location: AreaLocation | list[AreaLocation],
                            modeled_ms_area_location: AreaLocation | list[AreaLocation],
                            set_areas_globally: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """ Compare band ratios of real MS image data with modeled MS image data

        :param real_ms_area_location: AreaLocation or list[AreaLocation] objects describing the area that will be compared
        :param modeled_ms_square_mean: AreaLocation or list[AreaLocation] object describing the area that will be compared
        :param set_areas_globally: if True, provide only single AreaLocation, else list[AreaLocation] of length nbands
        :return: tuple(real_ms_ratios, modeled_ms_ratios)
        :raises ValueError: if sum of means of selected area is less than 1e-10
        """

        logging.info("[DataComparator] Preparing comparison...")

        if not real_ms_area_location or not modeled_ms_area_location:
            raise NoProvidedArea

        if self.ms_img_data.nbands != self.modeled_img_data.nbands:
            raise ImageDataIncompatible(
                f"Provided image data has incompatible number of bands ({self.ms_img_data.nbands} vs {self.modeled_img_data.nbands})"
            )

        if set_areas_globally:

            if isinstance(real_ms_area_location, list) or isinstance(modeled_ms_area_location, list):
                raise InvalidProvidedArea(
                    f"Expected single AreaLocation object for set_areas_globally = True, got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
                )

            real_ms_square_mean = ImageData.mean_spectrum_area(self.ms_img_data.img_data,
                                                               real_ms_area_location.as_tuple())
            modeled_ms_square_mean = ImageData.mean_spectrum_area(self.modeled_img_data.img_data,
                                                                  modeled_ms_area_location.as_tuple())

        else:
            if not isinstance(real_ms_area_location, list) or not isinstance(modeled_ms_area_location, list):
                raise InvalidProvidedArea(
                    f"Expected list of AreaLocation objects for set_areas_globally = False, got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
                )

            if len(real_ms_area_location) != self.ms_img_data.nbands:
                raise InvalidProvidedArea(
                    f"Provided area locations ({len(real_ms_area_location)}) does not match the number of bands ({self.ms_img_data.nbands})"
                )

            real_ms_square_mean = np.zeros(self.ms_img_data.nbands, dtype=np.float32)
            modeled_ms_square_mean = np.zeros(self.ms_img_data.nbands, dtype=np.float32)

            for band in range(self.ms_img_data.nbands):
                real_ms_square_mean[band] = ImageData.mean_spectrum_area(self.ms_img_data.img_data[:, :, band],
                                                                         real_ms_area_location[band].as_tuple())[0]
                modeled_ms_square_mean[band] = ImageData.mean_spectrum_area(
                    self.modeled_img_data.img_data[:, :, band], modeled_ms_area_location[band].as_tuple())[0]

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

    def calculate_spectral_angle_mapper(self, real_ms_ratios: np.ndarray, modeled_ms_ratios: np.ndarray) -> float:
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
