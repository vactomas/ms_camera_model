import numpy as np


def get_mean_spectrum_of_area(
        img,
        square_corner_coordinates: list[int] = [946, 349],
        pixels_per_dimension: int = 5) -> list[np.float64] | None:
    """ Get mean spectrum of a selected square area

    :param img: image data
    :param square_corner_coordinates: coordinates of an upper left corner of the square area
    :param pixels_per_dimension: number of pixels per dimension
    """

    pixel_spectrum = []
    NUM_OF_PIXEL_PER_DIMENSION = pixels_per_dimension

    for x_pixel in range(
            square_corner_coordinates[0],
            square_corner_coordinates[0] + NUM_OF_PIXEL_PER_DIMENSION):
        for y_pixel in range(
                square_corner_coordinates[1],
                square_corner_coordinates[1] + NUM_OF_PIXEL_PER_DIMENSION):
            pixel_spectrum.append(img[y_pixel, x_pixel, :].squeeze())

    mean_spectrum = np.mean(pixel_spectrum, axis=0)

    return list(mean_spectrum)


## Gemini
def vector_normalize(img: np.ndarray) -> np.ndarray:
    """ Vector normalizes hyperspectral image data.

    :param img: Hyperspectral image data (numpy array)
    :return: Vector-normalized image data
    """
    # Reshape the image data to a 2D array (pixels x bands)
    pixels = img.reshape(-1, img.shape[2])

    # Calculate the Euclidean norm (vector length) for each pixel's spectrum
    # The norm is the square root of the sum of the squares of all band values.
    norms = np.linalg.norm(pixels, axis=1)

    # Avoid division by zero for pixels with a norm of 0
    norms[norms == 0] = 1e-10

    # Divide each pixel's spectrum by its norm
    normalized_pixels = pixels / norms[:, np.newaxis]

    # Reshape the data back to the original image shape
    normalized_img = normalized_pixels.reshape(img.shape)

    return normalized_img
