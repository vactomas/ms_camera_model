'''
Multispectral Camera Model - Image Registrator
=============================================

Dataclasses and their methods for registering image data
'''

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2 as cv
import numpy as np
from cv2 import DMatch
from skimage import exposure

from ms_camera_model.errors import (
    ImageDataIncompatible,
    ImageRegistrationFailed,
    NoImageData,
)
from ms_camera_model.image_data import ImageData

logger = logging.getLogger(__name__)


class RegistrationAlgorithm(ABC):
    """ Define keypoint finding and matching strategy """

    @abstractmethod
    def find_keypoints(self, img: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """ Find keypoint using defined method

        :param img: image data array
        :param mask: mask array
        :return: keypoints and their descriptors
        """
        pass

    @abstractmethod
    def find_matches(self, des_ref: np.ndarray, des_src: np.ndarray) -> Sequence[Sequence[DMatch]]:
        """ Find matches using compatible matcher

        :param des_ref: descriptors for reference image
        :param des_src: descriptors for source image
        :return: matches
        """
        pass


class AkazeAlgorithm(RegistrationAlgorithm):
    """ AKAZE algorithm """

    def find_keypoints(self, img: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """ Find keypoint using AKAZE method

        :param img: image data array
        :param mask: mask array
        :return: keypoints and their descriptors
        """
        finder = cv.AKAZE_create()  # type: ignore
        kp, des = finder.detectAndCompute(img, mask)

        return kp, des

    def find_matches(self, des_ref: np.ndarray, des_src: np.ndarray) -> Sequence[Sequence[cv.DMatch]]:
        """ Find matches using compatible matcher

        :param des_ref: descriptors for reference image
        :param des_src: descriptors for source image
        :return: matches
        """

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des_ref, des_src, k=2)

        return matches


class OrbAlgorithm(RegistrationAlgorithm):
    """ ORB algorithm """

    def find_keypoints(self, img: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """ Find keypoint using ORB method

        :param img: image data array
        :param mask: mask array
        :return: keypoints and their descriptors
        """
        finder = cv.ORB_create()  # type: ignore
        kp, des = finder.detectAndCompute(img, mask)

        return kp, des

    def find_matches(self, des_ref: np.ndarray, des_src: np.ndarray) -> Sequence[Sequence[cv.DMatch]]:
        """ Find matches using compatible matcher

        :param des_ref: descriptors for reference image
        :param des_src: descriptors for source image
        :return: matches
        """

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des_ref, des_src, k=2)

        return matches


class SiftAlgorithm(RegistrationAlgorithm):
    """ SIFT algorithm """

    def find_keypoints(self, img: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """ Find keypoint using SIFT method

        :param img: image data array
        :param mask: mask array
        :return: keypoints and their descriptors
        """
        finder = cv.SIFT_create()  # type: ignore
        kp, des = finder.detectAndCompute(img, mask)

        return kp, des

    def find_matches(self, des_ref: np.ndarray, des_src: np.ndarray) -> Sequence[Sequence[cv.DMatch]]:
        """ Find matches using compatible matcher

        :param des_ref: descriptors for reference image
        :param des_src: descriptors for source image
        :return: matches
        """

        matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des_ref, des_src, k=2)

        return matches


def register_bands(reference_img: ImageData,
                   source_img: ImageData,
                   registration_strategy: RegistrationAlgorithm,
                   band_mask_paths_ref: list[str | None] | None = None,
                   band_mask_paths_src: list[str | None] | None = None) -> ImageData:
    """ Register img_data of one ImageData class instance against another

    :param reference_img: ImageData class instance that is being registered onto
    :param source_img: ImageData class instance that is being registered against the reference
    :param registration_strategy: image registration strategy
    :param band_mask_paths_ref: list of mask file paths for reference images
    :param band_mask_paths_src: list of mask file paths for source images
    :raises ImageDataIncompatible: if the number of bands differs in self and other
    """

    logger.info("[ImageRegistrator] Starting image registration...")

    if not reference_img or not source_img:
        raise NoImageData("No reference_img or source_img provided")

    if reference_img.nbands != source_img.nbands:
        raise ImageDataIncompatible(
            f"Provided ImageData instances have incompatible number of bands ({reference_img.nbands} vs {source_img.nbands})"
        )

    if source_img.img_data.size == 0:
        raise NoImageData("source_img.img_data is empty")

    if reference_img.img_data.size == 0:
        raise NoImageData("reference_img.img_data is empty")

    registered_img_data = np.zeros(
        (reference_img.img_data.shape[0], reference_img.img_data.shape[1], reference_img.img_data.shape[2]),
        dtype=np.float64)

    for i_band in range(reference_img.nbands):
        logger.info(f"[ImageRegistrator] Registering band {i_band} out of {reference_img.nbands}...")

        try:
            registered_img_data[:, :, i_band] = _register_band(reference_img, source_img, i_band, registration_strategy,
                                                               band_mask_paths_ref, band_mask_paths_src)
        except ImageRegistrationFailed as e:
            raise ImageRegistrationFailed(f"Image registration failed on band {i_band}") from e

    logger.info("[ImageRegistrator] Image registration completed")

    return reference_img._create_new_instance(registered_img_data)


def _register_band(reference_img: ImageData, source_img: ImageData, i_band: int,
                   registration_strategy: RegistrationAlgorithm, band_mask_paths_ref: list[str | None] | None,
                   band_mask_paths_src: list[str | None] | None) -> np.ndarray | None:
    """ Register single band """

    transformed_img = np.zeros((reference_img.img_data.shape[0], reference_img.img_data.shape[1]))

    img_src = source_img.img_data[:, :, i_band]
    img_ref = reference_img.img_data[:, :, i_band]

    if band_mask_paths_ref:
        mask_path = band_mask_paths_ref[i_band]
        if mask_path:
            logger.info(f"[ImageRegistrator] Using ref image mask: {band_mask_paths_ref[i_band]}")
            mask_ref = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        else:
            mask_ref = None
    else:
        mask_ref = None

    if band_mask_paths_src:
        mask_path = band_mask_paths_src[i_band]
        if mask_path:
            logger.info(f"[ImageRegistrator] Using src image mask: {band_mask_paths_src[i_band]}")
            mask_src = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        else:
            mask_src = None
    else:
        mask_src = None

    img_ref_exp_comp = exposure.equalize_hist(img_ref)
    img_src_exp_comp = exposure.equalize_hist(img_src)

    img_ref_calc = np.zeros_like(img_ref_exp_comp)
    img_src_calc = np.zeros_like(img_src_exp_comp)

    cv.normalize(img_ref_exp_comp, img_ref_calc, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    cv.normalize(img_src_exp_comp, img_src_calc, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

    kp_ref, des_ref = registration_strategy.find_keypoints(img_ref_calc, mask_ref)
    kp_src, des_src = registration_strategy.find_keypoints(img_src_calc, mask_src)

    matches = registration_strategy.find_matches(des_ref, des_src)
    logger.info(f"[ImageRegistrator] Found {len(matches)} matches while registering band {i_band}")

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    logger.info(f"[ImageRegistrator] Out of these, {len(good_matches)} are good")

    if len(good_matches) < 4:
        raise ImageRegistrationFailed("Not enough good matches to compute homography (minimum 4 required).")

    p1 = np.array([kp_ref[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    p2 = np.array([kp_src[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    homography, inliers = cv.findHomography(p2, p1, cv.RANSAC)

    if homography is None:
        raise ImageRegistrationFailed

    try:
        transformed_img[:, :] = cv.warpPerspective(img_src, homography, (img_ref.shape[1], img_ref.shape[0]))
        return transformed_img

    except Exception as e:
        raise ImageRegistrationFailed(f"Warping perspective for band {i_band} failed") from e
