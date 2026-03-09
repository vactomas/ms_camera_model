'''
Multispectral Camera Model - Data Comparison
============================================

* **Description:** Classes and their methods used for comparing real MS data with modeled data
* **Author:** Tomas Vacek
'''
import logging

import numpy as np

from ms_camera_model.errors import ImageDataIncompatible, InvalidProvidedArea
from ms_camera_model.image_data import ImageData

logger = logging.getLogger(__name__)


class DataComparator:

    def __init__(self, ms_img_data: ImageData, modeled_img_data: ImageData) -> None:

        self.ms_img_data: ImageData = ms_img_data
        self.modeled_img_data: ImageData = modeled_img_data

    def compare_band_ratios(self,
                            real_ms_area_location: list[int] | list[list[int]],
                            modeled_ms_area_location: list[int] | list[list[int]],
                            set_areas_globally: bool = True) -> tuple[list[int | float], list[int | float]]:
        """ Compare band ratios of real MS image data with modeled MS image data """

        logging.info("[DataComparator] Preparing comparison...")

        if set_areas_globally:

            if type(real_ms_area_location) is list[list[int]] or type(modeled_ms_area_location) is list[list[int]]:
                raise InvalidProvidedArea(
                    f"Expected area types list[int], got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
                )

            real_ms_square_mean = ImageData.mean_spectrum_area(self.ms_img_data, real_ms_area_location)
            modeled_ms_square_mean = ImageData.mean_spectrum_area(self.modeled_img_data, modeled_ms_area_location)

            real_ms_ratios = real_ms_square_mean / np.sum(real_ms_square_mean)
            modeled_ms_ratios = modeled_ms_square_mean / np.sum(modeled_ms_square_mean)

        else:
            if self.ms_img_data.nbands == self.modeled_img_data.nbands:

                if len(real_ms_area_location) != len(modeled_ms_area_location):
                    InvalidProvidedArea("Area location lists are not of the same size")

                if len(real_ms_area_location) != self.ms_img_data.nbands:
                    InvalidProvidedArea(
                        f"Provided area locations ({len(real_ms_area_location)}) does not match the number of bands ({self.ms_img_data.nbands})"
                    )

                real_ms_square_mean = np.zeros((self.ms_img_data.nbands))
                modeled_ms_square_mean = np.zeros((self.ms_img_data.nbands))

                for band in range(self.ms_img_data.nbands):

                    real_ms_square_mean[band] = ImageData.mean_spectrum_area(self.ms_img_data.img_data[:, :, band],
                                                                             real_ms_area_location[band])
                    modeled_ms_square_mean[band] = ImageData.mean_spectrum_area(
                        self.modeled_img_data.img_data[:, :, band], modeled_ms_area_location[band])

                real_ms_ratios = real_ms_square_mean / np.sum(real_ms_square_mean)
                modeled_ms_ratios = modeled_ms_square_mean / np.sum(modeled_ms_square_mean)

            else:
                raise ImageDataIncompatible

            logging.info(
                f"[DataComparator] SQR_mean: {real_ms_square_mean}, SUM: {np.sum(real_ms_square_mean)}, ratios: {real_ms_ratios}"
            )
            logging.info(
                f"[DataComparator] SQR_mean: {modeled_ms_square_mean}, SUM: {np.sum(modeled_ms_square_mean)}, ratios: {modeled_ms_ratios}"
            )

            return [real_ms_ratios, modeled_ms_ratios]

    @staticmethod
    def _find_mean_value_for_square(img_data_array: np.ndarray,
                                    x_coordinate: int,
                                    y_coordinate: int,
                                    square_size: int,
                                    single_band: bool = False) -> np.ndarray:
        """ Find mean value for a selected square 

        x_coordinate
        y_coordinate
        square_size
        :return: 
        """

        if single_band:

            values_of_selected_pixels = img_data_array[
                x_coordinate:x_coordinate + square_size,
                y_coordinate:y_coordinate + square_size,
            ]

        else:
            values_of_selected_pixels = img_data_array[x_coordinate:x_coordinate + square_size,
                                                       y_coordinate:y_coordinate + square_size, :]

        mean_values = values_of_selected_pixels.mean(axis=(0, 1))

        return mean_values
