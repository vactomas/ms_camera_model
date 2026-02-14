import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rotate, rescale

import logging
logger = logging.getLogger(__name__)


def apply_gaussian_lowpass(Z, sigma_cutoff):
    """
    Applies a Gaussian Low Pass Filter to the Frequency Domain Z.
    
    Args:
        Z: The complex frequency spectrum (from your code).
        sigma_cutoff: The 'width' of the filter. 
                      Lower value = Stronger blurring (removes more noise).
                      Higher value = Retains more high-freq detail.
    """
    rows, cols = Z.shape
    
    # 1. Generate Coordinate Grid centered at (0,0)
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x, y)
    
    # 2. Create the Gaussian Mask (Centered)
    # Formula: e^-( (x^2 + y^2) / (2 * sigma^2) )
    radius_sq = X**2 + Y**2
    gaussian_mask = np.exp(-radius_sq / (2 * sigma_cutoff**2))
    
    # 3. Shift the Mask to match FFT layout
    # Z has low frequencies (DC) at the corners (0,0). 
    # Our mask is currently centered. We must shift the center to the corners.
    gaussian_mask = np.fft.ifftshift(gaussian_mask)
    
    # 4. Apply Filter
    return Z * gaussian_mask


def calculate_shift(img_ref: np.ndarray, img_src: np.ndarray, sigma: float) -> list[float]:
    """ Calculate how much to shift the image """
    F_ref = np.fft.fft2(img_ref)
    F_src = np.fft.fft2(img_src)

    k = 0.0000001

    numerator = F_ref * np.conj(F_src)
    denominator = (np.abs(F_ref) + k) * (np.abs(F_src) + k)

    frequency_spectrum = numerator / denominator
    frequency_spectrum = apply_gaussian_lowpass(frequency_spectrum, sigma)

    C = np.fft.fftshift(np.abs(np.fft.ifft2(frequency_spectrum)))

    plt.imshow(C)
    #plt.show()

    y_center, x_center = np.unravel_index(np.argmax(C), C.shape)
        
    shift_x = x_center - img_ref.shape[1] // 2
    shift_y = y_center - img_ref.shape[0] // 2

    return [shift_y, shift_x]


def fit_shape(img, target_shape):
    """Crops or pads img to match target_shape center-wise."""
    logger.info("[FS] Fit shape")
    rows, cols = target_shape
    ch, cw = img.shape
    
    # Pad if too small
    diff_h, diff_w = rows - ch, cols - cw
    if diff_h > 0 or diff_w > 0:
        logger.info("[FS] Padding image")
        pad_h = max(0, diff_h)
        pad_w = max(0, diff_w)
        # Pad evenly on both sides
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), 
                           (pad_w//2, pad_w - pad_w//2)), mode='constant')
    
    # Crop if too big
    ch, cw = img.shape
    if ch > rows or cw > cols:
        logger.info("[FS] Cropping image")
        start_r = (ch - rows) // 2
        start_c = (cw - cols) // 2
        img = img[start_r:start_r+rows, start_c:start_c+cols]
        
    return img


def phase_corr_translation(img_ref: np.ndarray, img_src: np.ndarray, sigma: float) -> np.ndarray:
    """ Phase correlation for translation """

    logger.info("[T] Begin T phase correlation...")

    shift = calculate_shift(img_ref, img_src, sigma)

    logger.info(f"[T] Found shift {shift[0]}, {shift[1]}")

    translation_matrix = np.float32([ [1, 0, shift[0]], [0, 1, shift[1]] ])

    return cv.warpAffine(img_src, translation_matrix, (img_ref.shape[1], img_ref.shape[0]))


def phase_corr_rotation_scaling(img_ref: np.ndarray, img_src: np.ndarray, sigma: float) -> np.ndarray:
    """ Phase correlation for rotation and scaling """

    logger.info("[RS] Begin RS phase correlation...")

    mag_ref = np.abs(np.fft.fftshift(np.fft.fft2(img_ref)))
    mag_src = np.abs(np.fft.fftshift(np.fft.fft2(img_src)))

    shape = mag_ref.shape
    radius = shape[0] // 2

    img_ref_warped = warp_polar(mag_ref, radius=radius, scaling='log')
    img_src_warped = warp_polar(mag_src, radius=radius, scaling='log')

    shift = calculate_shift(img_ref_warped, img_src_warped, sigma)

    shiftr, shiftc = shift[:2]
    shift_angle = -(360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    logger.info(f"[RS] Found angle {shift_angle}, scale {shift_scale}")

    img_src_rotated = rotate(img_src, shift_angle)
    img_src_rotated_scaled = rescale(img_src_rotated, shift_scale)
    plt.imshow(img_src_rotated_scaled)
    #plt.show()


    logger.info(f"[RS] Outputting img with shape: {img_src_rotated_scaled.shape[0]}, {img_src_rotated_scaled.shape[1]}")

    return img_src_rotated_scaled


def phase_corr_translation_rotation_scaling(img_ref: np.ndarray, img_src: np.ndarray, sigma: float) -> np.ndarray:
    """ Phase correlation for rotation, scaling and translation """

    logger.info("[TRS] Begin TRS phase correlation...")

    img_src_rotated_scaled = phase_corr_rotation_scaling(img_ref, img_src, sigma)

    img_src_translated_rotated_scaled = phase_corr_translation(img_ref, fit_shape(img_src_rotated_scaled, img_ref.shape), sigma)

    return img_src_translated_rotated_scaled

