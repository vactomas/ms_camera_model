'''
=======================================================================================================================
- Name:         Multispectral camera model - Image data
- Description:  Dataclasses and their methods for image data
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations
from dataclasses import dataclass
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral
from skimage import img_as_float

from ms_camera_model.errors import ImageDataIncompatible, NoProvidedFilepaths, IncompatibleBandChoice, NoImageData

import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """ Image Data class 

    :param img_data: np.ndarray of image data with shape (rows, cols, bands)
    :param band_centers: list of floats storing the center of each band in the img_data array
    :param nbands: number of bands in the img_data array
    :param band_mask_paths: list of paths to masks for different bands (first path is for band 0, second for band 1, etc.)

    :method __add__: add two ImageData classes
    :method imshow: view the img_data as an RGB interpretation of selected bands or as a brightness plot
    :method vector_normalize: use vector normalization on the img_data
    :method mean_spectrum_area: calculate mean spectrum for a select area
    :method plot_area_spectrum: visualize the mean spectrum of an area
    :method register_images: register img_data of another ImageData class instance against this instance
    """

    img_data: np.ndarray | None = None
    band_centers: list[float] | None = None
    nbands: int | None = None

    def __add__(self, other: ImageData) -> ImageData:
        """ Addition of two ImageData classes 
        
        :param self:    self
        :param other:   other ImageData class
        :raises ImageDataIncompatible: when the provided class instances do not have matching band_centers and number of bands
        """

        logger.info("[ImageData] Performing ImageData addition...")

        if not self.nbands == other.nbands and not self.band_centers == other.band_centers:
            raise ImageDataIncompatible("ImageData objects used for addition are not compatible with each other")

        return ImageData(self.img_data + other.img_data, self.band_centers, self.nbands)

    def imshow(self, bands: list[int] = []) -> None:
        """ View image as an RGB interpretation of selected bands or as a brightness plot

        :param bands: list of band numbers, if the list is not provided, brightness plot will be used instead
        :raises IncompatibleBandChoice: if 'bands' is not [], len(bands) != 3 or the number is out-of-bounds
        """

        if len(bands) == 3:
            logger.info(
                f"[ImageData] Showing RGB interpretation from bands -> R:{bands[0]}, G:{bands[1]}, B:{bands[2]}")

            plot_band_list = [bands[0], bands[1], bands[2]]
            plot_data = self.img_data[:, :, (plot_band_list)]

            # Rescale values to range 0.0 - 1.0 for each band individually
            for i in range(len(bands)):
                plot_data[:, :, i] *= (1.0 / plot_data[:, :, i].max())

            plt.imshow(plot_data)
            plt.show()

        elif len(bands) == 1:
            logger.info(f"[ImageData] Showing band -> {bands[0]}")

            plot_band_list = [bands[0]]
            plot_data = self.img_data[:, :, (plot_band_list)]

            plot_data[:, :, 0] *= (1.0 / plot_data[:, :, 0].max())

            plt.imshow(plot_data)
            plt.show()

        elif bands == []:
            logger.info("[ImageData] Showing image as a brightness plot")

            non_empty_bands = self.img_data[(ceil(self.img_data.shape[0] / 2)),
                                            (ceil(self.img_data.shape[1] / 2)), :] > 1e-2
            bands = [i for i, x in enumerate(non_empty_bands) if x]
            plot_data = self.img_data[:, :, (bands)]
            plot_data = plot_data.sum(axis=2)
            plot_data /= len(bands)
            plot_data *= (1.0 / plot_data.max())

            plt.imshow(plot_data, cmap='gray', vmin=0.0, vmax=1.0)
            plt.show()

        else:
            logger.info(f"[ImageData] Wrong band choice was provided. Expected [] or len(bands) == 3, got {bands}")
            raise IncompatibleBandChoice

    def plot_area_spectrum(self, top_right_coordinate: list[int], pixels_per_dimension: int = 5) -> None:
        """ Plot spectrum of pixels 

        :param top_right_coordinate: list[x, y], where x and y correspond to pixel coordinates of the requested square
        :param pixels_per_dimension: pixels per side of the square
        """

        logger.info(
            f"[ImageData] Plotting mean spectrum of a {pixels_per_dimension} by {pixels_per_dimension} area starting at {top_right_coordinate}"
        )

        area_data = self.mean_spectrum_area(top_right_coordinate, pixels_per_dimension)
        plt.plot(self.band_centers, area_data, label="Spectral response")
        plt.xlabel("Band")
        plt.ylabel("Reflectance")
        plt.show()

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

        logger.info("[ImageData] Vector nomralization completed")

    def mean_spectrum_area(self,
                           corner_coords: list[int] = [946, 349],
                           pixels_per_dimension: int = 5) -> list[np.float64] | None:
        """ Get mean spectrum of a selected square area

        :param img: image data
        :param square_corner_coordinates: coordinates of an upper left corner of the square area
        :param pixels_per_dimension: number of pixels per dimension
        """

        logger.info(
            f"[ImageData] Calculating mean spectrum for area x: {corner_coords[0]}-{corner_coords[0] + pixels_per_dimension}, y: {corner_coords[1]}-{corner_coords[1] + pixels_per_dimension}..."
        )

        pixel_spectrum = self.img_data[corner_coords[0]:corner_coords[0] + pixels_per_dimension,
                                       corner_coords[1]:corner_coords[1] + pixels_per_dimension, :]
        mean_spectrum = pixel_spectrum.mean(axis=(0, 1))

        logger.info("[ImageData] Mean spectrum calculation completed")

        return list(mean_spectrum)

    def register_bands(self,
                       other: ImageData,
                       band_mask_paths_ref: list[str | None] | None = None,
                       band_mask_paths_src: list[str | None] | None = None) -> ImageData:
        """ Register img_data of another ImageData class instance against this instance 

        :param other: ImageData class instance that is being registered against this one
        :param band_mask_paths_ref: list of mask file paths for reference images
        :param band_mask_paths_src: list of mask file paths for source images
        :raises ImageDataIncompatible: if the number of bands differs in self and other
        """

        logger.info("[ImageData] Starting image registration...")

        if self.nbands != other.nbands:
            logger.info(
                f"[ImageData] Provided ImageData instances have incompatible number of bands ({self.nbands} vs {other.nbands})"
            )
            raise ImageDataIncompatible

        if self.img_data.size == 0:
            logger.info("[ImageData] self.img_data is empty")
            raise NoImageData
        elif other.img_data.size == 0:
            logger.info("[ImageData] other.img_data is empty")
            raise NoImageData

        registered_img_data = np.zeros((self.img_data.shape[0], self.img_data.shape[1], self.img_data.shape[2]))

        for i_band in range(self.nbands):
            logger.info(f"[ImageData] Registering band {i_band} out of {self.nbands}...")

            registered_img_data[:, :, i_band] = self._register_band(other, i_band, band_mask_paths_ref,
                                                                    band_mask_paths_src)

        logger.info(f"[ImageData] Image registration completed")

        return ImageData(registered_img_data, other.band_centers, other.nbands)

    def _register_band(self, other: ImageData, i_band: int, band_mask_paths_ref: list[str | None] | None,
                       band_mask_paths_src: list[str | None] | None) -> np.ndarray | None:
        """ Register single band """

        transformed_img = np.zeros((self.img_data.shape[0], self.img_data.shape[1]))

        img_ref = self.img_data[:, :, i_band]
        img_src = other.img_data[:, :, i_band]

        if band_mask_paths_ref:
            if band_mask_paths_ref[i_band]:
                logger.info(f"[ImageData] Using ref image mask: {band_mask_paths_ref[i_band]}")
                mask_ref = cv.imread(band_mask_paths_ref[i_band], cv.IMREAD_GRAYSCALE)
            else:
                mask_ref = None
        else:
            mask_ref = None

        if band_mask_paths_src:
            if band_mask_paths_src[i_band]:
                logger.info(f"[ImageData] Using src image mask: {band_mask_paths_src[i_band]}")
                mask_src = cv.imread(band_mask_paths_src[i_band], cv.IMREAD_GRAYSCALE)
            else:
                mask_src = None
        else:
            mask_src = None

        img_ref = cv.normalize(img_ref, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
        img_src = cv.normalize(img_src, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

        finder = cv.ORB_create()
        kp_ref, des_ref = finder.detectAndCompute(img_ref, mask_ref)
        kp_src, des_src = finder.detectAndCompute(img_src, mask_src)
        logger.info(f"[ImageData] Found {len(kp_ref)} points in reference image and {len(kp_src)} in source image.")

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_ref, des_src)
        logger.info(f"[ImageData] Found {len(matches)} matches while registering band {i_band}")

        img_matches = cv.drawMatches(img_ref,
                                     kp_ref,
                                     img_src,
                                     kp_src,
                                     matches,
                                     None,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 5))
        plt.imshow(img_matches)
        plt.title('Feature Matches')
        plt.show()

        p1 = np.zeros((len(matches), 2))
        p2 = np.zeros((len(matches), 2))

        for i_point in range(len(matches)):
            p1[i_point, :] = kp_ref[matches[i_point].queryIdx].pt
            p2[i_point, :] = kp_src[matches[i_point].trainIdx].pt

        homography, _ = cv.findHomography(p2, p1, cv.RANSAC)

        if homography is None:
            logger.warning(f"[ImageData] Couldn't find homography for band {i_band}, skipping...")
            return transformed_img

        try:
            transformed_img[:, :] = cv.warpPerspective(img_src, homography, (img_ref.shape[1], img_ref.shape[0]))
            return transformed_img

        except Exception as ex:
            logger.critical(f"[ImageData] Warping perspective for band {i_band} failed: {ex}")
            return None


@dataclass
class MultispectralImageData(ImageData):
    """ Multispectral Image Data - loaded from real MS camera 

    :method load_ms_imgs: Load multispectral image data into img_data as np.ndarray
    """

    def import_ms_imgs(self, filepaths: list[str]) -> None:
        """ Import multispectral images as a np.ndarray 

        :param filepaths: list of filepaths to the multispectral images, their order determines order in the final array
        :raises NoProvidedFilepaths: when the filepaths param is empty
        """

        logger.info(f"[ImageData] Beginning import of multispectral images...")

        num_of_images = len(filepaths)

        if filepaths is []:
            raise NoProvidedFilepaths

        for i_img in range(num_of_images):
            logger.info(f"[ImageData] Importing multispectral image {filepaths[i_img]}...")

            img = cv.imread(filepaths[i_img], cv.IMREAD_UNCHANGED)

            if i_img == 0:
                self.img_data = np.zeros((img.shape[0], img.shape[1], num_of_images))

            self.img_data[:, :, i_img] = img_as_float(img)

        self.nbands = num_of_images

        logger.info("[ImageData] Import of multispectral images completed")


@dataclass
class HyperspectralImageData(ImageData):
    """ Hyperspectral Image Data - imported from hyperspectral data file 

    :method import_hs_img: Import hyperspectral cube into img_data, band_centers and nbands
    """

    def import_hs_img(self, filepath: str) -> None:
        """ Import hyperspectral cube as ImageData class instance 

        :param filepath: path to the hyperspectral image file
        """

        logger.info(f"[ImageData] Beginning import of hyperspectral file {filepath}...")

        img = spectral.open_image(filepath)

        self.img_data = img.load()
        self.band_centers = img.bands.centers
        self.nbands = img.nbands

        logger.info("[ImageData] Hyperspectral image import completed")
