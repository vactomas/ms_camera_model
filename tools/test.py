import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_window(shape):
    """Creates a 2D Hanning window to reduce FFT edge artifacts."""
    rows, cols = shape
    return np.outer(np.hanning(rows), np.hanning(cols))

def match_shape(img_to_resize, target_shape):
    """
    Crops or pads an image to match the target_shape exactly.
    Essential after rescaling, as dimensions will change.
    """
    t_rows, t_cols = target_shape
    s_rows, s_cols = img_to_resize.shape
    
    # 1. Pad if too small
    pad_h = max(0, t_rows - s_rows)
    pad_w = max(0, t_cols - s_cols)
    if pad_h > 0 or pad_w > 0:
        img_to_resize = np.pad(img_to_resize, 
                               ((pad_h//2, pad_h - pad_h//2), 
                                (pad_w//2, pad_w - pad_w//2)), 
                               mode='constant')

    # 2. Crop if too big
    s_rows, s_cols = img_to_resize.shape # Update shape after padding
    if s_rows > t_rows or s_cols > t_cols:
        start_r = (s_rows - t_rows) // 2
        start_c = (s_cols - t_cols) // 2
        img_to_resize = img_to_resize[start_r:start_r+t_rows, start_c:start_c+t_cols]
        
    return img_to_resize

def apply_gaussian_lowpass(Z, sigma_cutoff):
    rows, cols = Z.shape
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x, y)
    
    radius_sq = X**2 + Y**2
    gaussian_mask = np.exp(-radius_sq / (2 * sigma_cutoff**2))
    gaussian_mask = np.fft.ifftshift(gaussian_mask)
    
    return Z * gaussian_mask

def calculate_shift(F_ref, F_src, sigma, p=1e-5, q=1e-5):
    """
    Core Phase Correlation Logic (Your Z formula).
    Expects Frequency Domain inputs (F_ref, F_src).
    """
    # Your Custom Formula
    numerator = F_ref * np.conj(F_src)
    denominator = (np.abs(F_ref) + p) * (np.abs(F_src) + q)
    
    Z = numerator / denominator
    
    # Low Pass Filter
    Z_filtered = apply_gaussian_lowpass(Z, sigma)
    
    # Inverse FFT to get spatial correlation surface
    C = np.abs(np.fft.ifft2(Z_filtered))
    C = np.fft.fftshift(C) # Move peak to center
    
    # Find Peak
    y_center, x_center = np.unravel_index(np.argmax(C), C.shape)
    
    shift_y = y_center - C.shape[0] // 2
    shift_x = x_center - C.shape[1] // 2
    
    return shift_y, shift_x, C

def solve_rotation_scaling_robust(img_ref, img_src, sigma):
    logger.info("[RS] Calculating Rotation and Scale...")
    
    # 1. Windowing (Standard)
    win = create_window(img_ref.shape)
    
    # 2. FFT Magnitude with HIGH PASS FILTER
    # Real images have massive energy at the center (DC). We must block it 
    # so the correlation locks onto edges/structures, not just average brightness.
    F_ref = np.fft.fftshift(np.fft.fft2(img_ref * win))
    F_src = np.fft.fftshift(np.fft.fft2(img_src * win))
    
    M_ref = np.abs(F_ref)
    M_src = np.abs(F_src)
    
    # --- NEW: High Pass Filter (Block the center) ---
    # Create a small circular mask at the center to block low freqs
    rows, cols = M_ref.shape
    cy, cx = rows // 2, cols // 2
    r_mask = 5 # Block 5 pixels radius
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    mask = x*x + y*y > r_mask*r_mask
    
    M_ref = M_ref * mask
    M_src = M_src * mask
    # ------------------------------------------------

    # Log scaling helps equalize dynamic range for correlation
    M_ref = np.log(M_ref + 1)
    M_src = np.log(M_src + 1)

    # 3. Log-Polar Transform
    radius = min(M_ref.shape) // 2
    pol_ref = warp_polar(M_ref, radius=radius, scaling='log')
    pol_src = warp_polar(M_src, radius=radius, scaling='log')
    
    # 4. Phase Correlation
    F_pol_ref = np.fft.fft2(pol_ref)
    F_pol_src = np.fft.fft2(pol_src)
    
    shift_r, shift_c, _ = calculate_shift(F_pol_ref, F_pol_src, sigma)
    
    # 5. Recover Parameters
    angle_est = -(shift_r * 360.0 / pol_ref.shape[0])
    k_log = pol_ref.shape[1] / np.log(radius)
    scale_est = 1.0 / np.exp(shift_c / k_log)
    
    return angle_est, scale_est

def full_registration_pipeline(img_ref, img_src, sigma=5.0):
    img_ref = img_as_float(img_ref)
    img_src = img_as_float(img_src)
    
    # 1. Initial Guess (Rotation/Scale)
    angle, scale = solve_rotation_scaling_robust(img_ref, img_src, sigma)
    
    # --- NEW: Fix 180-degree Ambiguity ---
    # Because FFT magnitude is symmetric, we might be off by exactly 180 degrees.
    # We will test both `angle` and `angle + 180` and see which gives a better correlation peak.
    
    candidates = [angle, angle + 180]
    best_peak = -1
    best_angle = angle
    best_scale = scale
    best_final_img = None
    best_shift = (0,0)

    logger.info(f"[TRS] Checking 180-degree ambiguity for angles: {candidates}")

    for ang in candidates:
        # Apply Transform
        # Note: We must re-match shape every time we rotate/scale
        temp_rs = rotate(img_src, ang, resize=False)
        temp_rs = rescale(temp_rs, scale, channel_axis=None)
        temp_rs = match_shape(temp_rs, img_ref.shape)
        
        # Check Translation & Peak Height
        # We assume the "Winner" is the one with the stronger correlation peak
        dy, dx, correlation_surface = solve_translation(img_ref, temp_rs, sigma)
        peak_val = np.max(correlation_surface)
        
        if peak_val > best_peak:
            best_peak = peak_val
            best_angle = ang
            best_final_img = temp_rs
            best_shift = (dy, dx)
    
    # -------------------------------------

    logger.info(f"[TRS] Winner: Angle {best_angle:.2f}, Peak {best_peak:.4f}")

    # Apply final translation to the winner
    dy, dx = best_shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_final = cv.warpAffine(np.float32(best_final_img), M, (img_ref.shape[1], img_ref.shape[0]))
    
    return img_final, best_shift, best_angle, scale

def solve_translation(img_ref, img_src, sigma):
    """
    Standard Phase Correlation for Translation.
    """
    logger.info("[T] Calculating Translation...")
    
    win = create_window(img_ref.shape)
    F_ref = np.fft.fft2(img_ref * win)
    F_src = np.fft.fft2(img_src * win)
    
    dy, dx, C = calculate_shift(F_ref, F_src, sigma)
    
    logger.info(f"[T] Detected Shift: y={dy}, x={dx}")
    return dy, dx, C


# ==========================================
# TEST AREA (Synthetic Data)
# ==========================================
if __name__ == "__main__":
    # 1. Create Synthetic Image (A square with some noise)
    img_ref = cv.imread("../img_data/NewData/IMG_0000_1.tif", cv.IMREAD_GRAYSCALE)
    img_src = cv.imread("../img_data/NewData/IMG_0000_2.tif", cv.IMREAD_GRAYSCALE)
    img_src_2 = cv.imread("../img_data/NewData/IMG_0000_3.tif", cv.IMREAD_GRAYSCALE)
    img_src_3 = cv.imread("../img_data/NewData/IMG_0000_4.tif", cv.IMREAD_GRAYSCALE)
    
    # 3. Run Pipeline
    registered_img, (dy, dx), d_ang, d_scale = full_registration_pipeline(img_ref, img_src, sigma=img_ref.shape[0]*0.1)
    registered_img_2, *_ = full_registration_pipeline(img_ref, img_src_2, sigma=img_ref.shape[0]*0.1)
    registered_img_3, *_ = full_registration_pipeline(img_ref, img_src_3, sigma=img_ref.shape[0]*0.1)
    
    final_img = np.zeros((img_ref.shape[0], img_ref.shape[1], 3))
    # final_img[:, :, 0] = img_ref
    final_img[:, :, 0] = registered_img
    final_img[:, :, 1] = registered_img_2
    final_img[:, :, 2] = registered_img_3

    for img in range(3):
        plt.imshow(final_img[:, :, img])
        plt.show()

    plt.imshow(final_img)
    plt.show()
