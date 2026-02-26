'''
=======================================================================================================================
- Name:         Multispectral camera model - Data comparison
- Description:  Dataclasses and their methods for filters and sensors
- Author:       Tomas Vacek
=======================================================================================================================
'''

import logging

import matplotlib.pyplot as plt
import numpy as np

from ms_camera_model.image_data import ImageData

logger = logging.getLogger(__name__)


class DataComparator:

    def __init__(self, ms_img_data: ImageData, modeled_img_data: ImageData) -> None:

        self.ms_img_data: ImageData = ms_img_data
        self.modeled_img_data: ImageData = modeled_img_data

    def compare_band_ratios(self, set_areas_globaly: bool = True) -> tuple[list[float], list[float]]:
        """ Compare band ratios of real MS image data with modeled MS image data """

        if set_areas_globaly:
            logging.info("[DataComparator] Select area for comparison")

            plt.ion()

            self.ms_img_data.imshow()

            x_coordinate = int(input("Type in x coordinate of upper left square corner: ").strip())
            y_coordinate = int(input("Type in y coordinate of upper left square corner: ").strip())
            square_size = int(input("Type in edge size (defaults to 10): ") or "10")

            logging.info(f"[DataComparator] {self.ms_img_data.img_data.shape}")
            logging.info(f"[DataComparator] {self.modeled_img_data.img_data.shape}")

            real_ms_square_mean = self._find_mean_value_for_square(self.ms_img_data.img_data, y_coordinate,
                                                                   x_coordinate, square_size)
            modeled_ms_square_mean = self._find_mean_value_for_square(self.modeled_img_data.img_data, y_coordinate,
                                                                      x_coordinate, square_size)

            real_ms_ratios = real_ms_square_mean / np.sum(real_ms_square_mean)
            modeled_ms_ratios = modeled_ms_square_mean / np.sum(modeled_ms_square_mean)

            logging.info(
                f"[DataComparator] SQR_mean: {real_ms_square_mean}, SUM: {np.sum(real_ms_square_mean)}, ratios: {real_ms_ratios}"
            )
            logging.info(
                f"[DataComparator] SQR_mean: {modeled_ms_square_mean}, SUM: {np.sum(modeled_ms_square_mean)}, ratios: {modeled_ms_ratios}"
            )

            return [real_ms_ratios, modeled_ms_ratios]

        else:
            logging.info("[DataComparator] Select areas for comparison for each band")

            if self.ms_img_data.nbands == self.modeled_img_data.nbands:

                real_ms_square_mean = np.zeros((self.ms_img_data.nbands))
                modeled_ms_square_mean = np.zeros((self.ms_img_data.nbands))

                plt.ion()

                for band in range(self.ms_img_data.nbands):

                    self.ms_img_data.imshow([band])

                    x_coordinate = int(input("Type in x coordinate of upper left square corner: ").strip())
                    y_coordinate = int(input("Type in y coordinate of upper left square corner: ").strip())
                    square_size = int(input("Type in edge size (defaults to 10): ") or "10")

                    real_ms_square_mean[band] = self._find_mean_value_for_square(self.ms_img_data.img_data[:, :, band],
                                                                                 y_coordinate,
                                                                                 x_coordinate,
                                                                                 square_size,
                                                                                 single_band=True)

                    self.modeled_img_data.imshow([band])

                    x_coordinate = int(input("Type in x coordinate of upper left square corner: ").strip())
                    y_coordinate = int(input("Type in y coordinate of upper left square corner: ").strip())
                    square_size = int(input("Type in edge size (defaults to 10): ") or "10")

                    modeled_ms_square_mean[band] = self._find_mean_value_for_square(
                        self.modeled_img_data.img_data[:, :, band],
                        y_coordinate,
                        x_coordinate,
                        square_size,
                        single_band=True)

                plt.ioff()

                real_ms_ratios = real_ms_square_mean / np.sum(real_ms_square_mean)
                modeled_ms_ratios = modeled_ms_square_mean / np.sum(modeled_ms_square_mean)

                logging.info(
                    f"[DataComparator] SQR_mean: {real_ms_square_mean}, SUM: {np.sum(real_ms_square_mean)}, ratios: {real_ms_ratios}"
                )
                logging.info(
                    f"[DataComparator] SQR_mean: {modeled_ms_square_mean}, SUM: {np.sum(modeled_ms_square_mean)}, ratios: {modeled_ms_ratios}"
                )

                return [real_ms_ratios, modeled_ms_ratios]

            else:
                raise ImageDataIncompatible

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
