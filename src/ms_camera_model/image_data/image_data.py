'''
=======================================================================================================================
- Name:         Multispectral camera model - Image data
- Description:  Dataclasses and their methods for image data
- Author:       Tomas Vacek
=======================================================================================================================
'''

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import micasense.metadata as metadata
import micasense.utils as msutils
import numpy as np
import spectral
from skimage import exposure, img_as_float
from skimage.transform import PiecewiseAffineTransform, warp

from ms_camera_model.errors import (
    ImageDataIncompatible,
    IncompatibleBandChoice,
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
    band_centers: list[float] | None = None
    nbands: int | None = None

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
    def mean_spectrum_area(img: np.ndarray, corner_coords: list[int]) -> np.ndarray:
        """ Calculate mean over spectral bands for select area

        :param img: image data
        :param corner_coords: coordinates of corners of the area in format [ulx, uly, lrx, lry]
        """

        logger.info(
            f"[ImageData] Calculating mean for area x: {corner_coords[0]}-{corner_coords[2]}, y: {corner_coords[1]}-{corner_coords[3]}..."
        )

        pixel_spectrum = img[corner_coords[1]:corner_coords[3], corner_coords[0]:corner_coords[2], :]
        mean_spectrum = pixel_spectrum.mean(axis=(0, 1))

        logger.info("[ImageData] Mean calculation for selected area completed")

        return mean_spectrum

    def register_bands(self,
                       other: ImageData | int,
                       band_mask_paths_ref: list[str | None] | None = None,
                       band_mask_paths_src: list[str | None] | None = None) -> ImageData | None:
        """ Register img_data of another ImageData class instance against this instance 

        :param other: ImageData class instance that is being registered against this one
        :param band_mask_paths_ref: list of mask file paths for reference images
        :param band_mask_paths_src: list of mask file paths for source images
        :raises ImageDataIncompatible: if the number of bands differs in self and other
        """

        logger.info("[ImageData] Starting image registration...")

        if type(other) is not int:

            if self.nbands != other.nbands:
                logger.info(
                    f"[ImageData] Provided ImageData instances have incompatible number of bands ({self.nbands} vs {other.nbands})"
                )
                raise ImageDataIncompatible

            if other.img_data.size == 0:
                logger.info("[ImageData] other.img_data is empty")
                raise NoImageData

        if self.img_data.size == 0:
            logger.info("[ImageData] self.img_data is empty")
            raise NoImageData

        registered_img_data = np.zeros((self.img_data.shape[0], self.img_data.shape[1], self.img_data.shape[2]),
                                       dtype=np.float64)

        for i_band in range(self.nbands):
            logger.info(f"[ImageData] Registering band {i_band} out of {self.nbands}...")

            registered_img_data[:, :, i_band] = self._register_band(other, i_band, band_mask_paths_ref,
                                                                    band_mask_paths_src)

        logger.info(f"[ImageData] Image registration completed")

        if type(other) is int:
            return ImageData(registered_img_data, self.band_centers, self.nbands)
        else:
            return ImageData(registered_img_data, other.band_centers, other.nbands)

    def _register_band(self, other: ImageData | int, i_band: int, band_mask_paths_ref: list[str | None] | None,
                       band_mask_paths_src: list[str | None] | None) -> np.ndarray | None:
        """ Register single band """

        transformed_img = np.zeros((self.img_data.shape[0], self.img_data.shape[1]), dtype=np.float64)

        img_ref = self.img_data[:, :, i_band]

        if type(other) is int:
            img_src = self.img_data[:, :, other]
        else:
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

        img_ref_exp_comp = exposure.equalize_hist(img_ref)
        img_src_exp_comp = exposure.equalize_hist(img_src)
        # img_ref_exp_comp = exposure.rescale_intensity(img_ref_exp_comp, in_range=(0, 100), out_range=(0, 1))
        # img_src_exp_comp = exposure.rescale_intensity(img_src_exp_comp, in_range=(0, 100), out_range=(0, 1))

        img_ref_calc = cv.normalize(img_ref_exp_comp, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
        img_src_calc = cv.normalize(img_src_exp_comp, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

        finder = cv.AKAZE_create(threshold=0.000001)
        kp_ref, des_ref = finder.detectAndCompute(img_ref_calc, mask_ref)
        kp_src, des_src = finder.detectAndCompute(img_src_calc, mask_src)
        logger.info(f"[ImageData] Found {len(kp_ref)} points in reference image and {len(kp_src)} in source image.")

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des_ref, des_src, k=2)
        logger.info(f"[ImageData] Found {len(matches)} matches while registering band {i_band}")

        good_matches = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        logger.info(f"[ImageData] Out of these, {len(good_matches)} are good")

        p1 = np.float32([kp_ref[m.queryIdx].pt
                         for m in good_matches])  #.reshape(-1, 1, 2)
        p2 = np.float32([kp_src[m.trainIdx].pt
                         for m in good_matches])  #.reshape(-1, 1, 2)

        img_matches = cv.drawMatches(img_ref_calc,
                                     kp_ref,
                                     img_src_calc,
                                     kp_src,
                                     good_matches,
                                     None,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 5))
        plt.imshow(img_matches)
        plt.title('Feature Matches')
        plt.show()

        homography, inliers = cv.findHomography(p2, p1, cv.RANSAC, 2.5)

        if homography is None or inliers is None:
            logger.warning(f"[ImageData] Couldn't find homography for band {i_band}, skipping...")
            return transformed_img

        try:
            # transformed_img[:, :] = cv.warpPerspective(img_src, homography, (img_ref.shape[1], img_ref.shape[0]))
            p1 = p1[inliers.flatten() == 1]
            p2 = p2[inliers.flatten() == 1]

            tform = PiecewiseAffineTransform()
            tform.estimate(p1, p2)

            rows, cols = img_ref.shape[:2]

            transformed_img = warp(img_src, tform, output_shape=(rows, cols), order=1, mode='edge')

            if img_ref.dtype == np.uint8:
                transformed_img = (transformed_img * 255).astype(np.uint8)
            elif img_ref.dtype == np.uint16:
                transformed_img = (transformed_img * 65535).astype(np.uint16)

            return transformed_img

        except Exception as ex:
            logger.critical(f"[ImageData] Warping perspective for band {i_band} failed: {ex}")
            return None

    def normalize_img_data(self) -> None:
        """ Normalize image data """

        max_value = np.max(self.img_data)

        self.img_data = self.img_data / max_value


@dataclass
class MultispectralImageData(ImageData):
    """ Multispectral Image Data - loaded from real MS camera 

    :method import_altum_pt_ms_imgs: method for import of images from MicaSense Altum-PT camera
    :method import_ms_imgs: method for import of generic multispectral images
    :method check_filepaths: method which checks that filepaths are a non-empty list and are strings
    """

    @classmethod
    def import_altum_pt_ms_imgs(cls, filepaths: list[str], panel_calibration: dict[str, float],
                                panel_location: list[list[int]]) -> MultispectralImageData:
        """ Import and pre-process Altum PT images 

        :param filepaths: list of filepaths to the multispectral images, their order determines order in the final array
        :param panel_calibration: panel_calibration data of used MicaSense CRP
        :param panel_location: panel_location information format [[ulx, uly, lrx, lry], [ulx, ...]]
        :raises NoProvidedArea: when the provided area of the CRP is None or empty
        :raises Exception: when the length of filepaths and panel_locations doesn't match
        :raises TypeError: when provided paths aren't in list
        :raises TypeError: when provided paths aren't strings
        :raises NoProvidedFilepaths: when there are no provided filepaths
        """

        logger.info("[ImageData] Beginning import of multispectral images...")

        if not len(panel_location) == len(filepaths):
            raise Exception("Number of image filepaths doesn't match number of calibration panel coordinates")

        if not panel_location or panel_location is None:
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

            coordinates = panel_location[i_img]
            mean_radiance = ImageData.mean_spectrum_area(img_raw, coordinates)

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
    def import_hs_img(cls, filepath: str) -> HyperspectralImageData:
        """ Import hyperspectral cube as ImageData class instance 

        :param filepath: path to the hyperspectral image file
        :raises NoImageData: when spectral fails to load the image
        """

        logger.info(f"[ImageData] Beginning import of hyperspectral file {filepath}...")

        try:
            img = spectral.open_image(filepath)

        except Exception as e:
            logger.error(f"[ImageData] Loading hyperspectral data from file {filepath} ended with error {e}")
            raise NoImageData from e

        img_data = img.load()
        band_centers = img.bands.centers
        nbands = img.nbands

        logger.info("[ImageData] Hyperspectral image import completed")

        return HyperspectralImageData(img_data, band_centers, nbands)

    def _perform_radiometric_calibration(self, filepath: str) -> None:
        """ Perform radiometric calibration based on calibration plate with known albedo

        :param filepath: path to the wavelength-albedo file
        """
