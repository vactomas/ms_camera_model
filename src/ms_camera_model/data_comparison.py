"""
Multispectral Camera Model - Data Comparison
============================================

* **Description:** Classes and their methods used for comparing real MS data with modeled data
* **Author:** Tomas Vacek
* **Year:** 2026
* **License:** MIT License
"""
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
    :raises InvalidProvidedArea: if provided area locations are lists
    """

    logger.info("[DataComparator] Preparing comparison...")

    _check_before_comparison(real_ms_image_data, modeled_ms_image_data, real_ms_area_location, modeled_ms_area_location)

    if isinstance(real_ms_area_location, list) or isinstance(modeled_ms_area_location, list):
        raise InvalidProvidedArea(
            f"Expected single AreaLocation object, got {type(real_ms_area_location)} and {type(modeled_ms_area_location)}"
        )

    real_ms_square_mean = ImageData.mean_spectrum_area(real_ms_image_data.img_data, real_ms_area_location.as_tuple())
    modeled_ms_square_mean = ImageData.mean_spectrum_area(modeled_ms_image_data.img_data,
                                                          modeled_ms_area_location.as_tuple())

    real_ms_ratios, modeled_ms_ratios = _calculate_band_ratios(real_ms_square_mean, modeled_ms_square_mean)

    return real_ms_ratios, modeled_ms_ratios


def compare_band_ratios_per_band(real_ms_image_data: MultispectralImageData,
                                 modeled_ms_image_data: ModeledMultispectralImageData,
                                 real_ms_area_location: list[AreaLocation],
                                 modeled_ms_area_location: list[AreaLocation]) -> tuple[np.ndarray, np.ndarray]:
    """ Compare band ratios of real MS image data with modeled MS image data, area defined per band

    :param real_ms_image_data: MS image data from a real camera
    :param modeled_ms_image_data: modeled MS image data
    :param real_ms_area_location: list[AreaLocation] objects describing the area that will be compared
    :param modeled_ms_area_location: list[AreaLocation] object describing the area that will be compared
    :return: tuple(real_ms_ratios, modeled_ms_ratios)
    :raises ValueError: if sum of means of selected area is less than 1e-10
    :raises InvalidProvidedArea: if provided area locations are not lists
    :raises InvalidProvidedArea: if provided area locations are not of necessary length
    """

    _check_before_comparison(real_ms_image_data, modeled_ms_image_data, real_ms_area_location, modeled_ms_area_location)

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

    real_ms_ratios, modeled_ms_ratios = _calculate_band_ratios(real_ms_square_mean, modeled_ms_square_mean)

    return real_ms_ratios, modeled_ms_ratios


def _check_before_comparison(real_ms_image_data: MultispectralImageData,
                             modeled_ms_image_data: ModeledMultispectralImageData,
                             real_ms_area_location: AreaLocation | list[AreaLocation],
                             modeled_ms_area_location: AreaLocation | list[AreaLocation]) -> None:
    """ Check if ImageData child class instances are compatible with each other and areas are provided 

    :param real_ms_image_data: MultispectralImageData instance
    :param modeled_ms_image_data: ModeledMultispectralImageData instance
    :param real_ms_area_location: AreaLocation or list of AreaLocation instances
    :param modeled_ms_area_location: AreaLocation or list of AreaLocation instances
    """

    if not real_ms_area_location or not modeled_ms_area_location:
        raise NoProvidedArea

    if real_ms_image_data.nbands != modeled_ms_image_data.nbands:
        raise ImageDataIncompatible(
            f"Provided image data has incompatible number of bands ({real_ms_image_data.nbands} vs {modeled_ms_image_data.nbands})"
        )


def _calculate_band_ratios(real_ms_square_mean: np.ndarray,
                           modeled_ms_square_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Calculate band ratios 

    :param real_ms_square_mean: 1D array of means for defined area on real camera image data
    :param modeled_ms_square_mean: 1D array of means for defined area on modeled image data
    :return: tuple of real_ms_ratios and modeled_ms_ratios
    :raises ValueError: if the denominator for division is less than 1e-10
    """
    sum_real_ms_mean = np.sum(real_ms_square_mean)
    sum_modeled_ms_mean = np.sum(modeled_ms_square_mean)

    if sum_real_ms_mean < 1e-10 or sum_modeled_ms_mean < 1e-10:
        raise ValueError("Denominator for next operation is 0 or close to 0")

    real_ms_ratios = real_ms_square_mean / sum_real_ms_mean
    modeled_ms_ratios = modeled_ms_square_mean / sum_modeled_ms_mean

    logger.info(
        f"[DataComparator] SQR_mean: {real_ms_square_mean}, SUM: {np.sum(real_ms_square_mean)}, ratios: {real_ms_ratios}"
    )
    logger.info(
        f"[DataComparator] SQR_mean: {modeled_ms_square_mean}, SUM: {np.sum(modeled_ms_square_mean)}, ratios: {modeled_ms_ratios}"
    )

    return real_ms_ratios, modeled_ms_ratios


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


def calculate_michelson_contrast(image_data: ModeledMultispectralImageData, band: int) -> float:
    """ Calculate Michelson contrast for a single band

    :param image_data: ModeledMultispectralImageData class instance
    :param band: band number as an integer
    :return: Michelson contrast as a float
    """
    logger.info("[DataComparator] Calculating Michelson contrast...")

    img_data = image_data.img_data

    max_val = img_data[:, :, band].max()
    min_val = img_data[:, :, band].min()

    michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-10)

    return michelson_contrast


def calculate_weber_contrast(image_data: ModeledMultispectralImageData, band: int, area_of_interest: AreaLocation,
                             background_area: AreaLocation) -> float:
    """ Calculate Weber contrast for a single band and defined area

    :param image_data: ModeledMultispectralImageData class instance
    :param band: band number as an integer
    :param area_of_interest: AreaLocation instance with coordinates of the area of interest
    :param background_area: AreaLocation instance with coordinates of the background area
    :return: Weber contrast as a float
    """
    logger.info("[DataComparator] Calculating Weber contrast...")

    img_data = image_data.img_data

    mean_target = ImageData.mean_spectrum_area(img_data[:, :, band], area_of_interest.as_tuple())
    mean_background = ImageData.mean_spectrum_area(img_data[:, :, band], background_area.as_tuple())

    weber_contrast = (mean_target - mean_background) / (mean_background + 1e-10)

    return weber_contrast


def calculate_ndi(image_data: ModeledMultispectralImageData, band_1: int, band_2: int) -> np.ndarray:
    """ Calculate Normalised Difference Index (NDI) for selected areas

    :param image_data: ModeledMultispectralImageData class instance
    :param band_1: band number as an integer
    :param band_2: band number as an integer
    :return: 1D array of Normalised Difference Index values per band
    """
    logger.info("[DataComparator] Calculating Normalised Difference Index (NDI)...")

    img_data = image_data.img_data

    nominator = img_data[:, :, band_1] - img_data[:, :, band_2]
    denominator = img_data[:, :, band_1] + img_data[:, :, band_2] + 1e-10

    ndi = np.divide(nominator, denominator)

    return ndi
