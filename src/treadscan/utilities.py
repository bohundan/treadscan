"""
This module contains various useful methods that can be used in multiple different places.
"""

from math import sin, cos, radians

import cv2
import numpy as np


class Ellipse:
    """
    Class which defines an ellipse.

    Attributes
    ----------
    cx : int
        X coordinate of center.

    cy : int
        Y coordinate of center.

    width : int
        Width of ellipse (size across the X axis if angle is 0).

    height : int
        Height of ellipse (size across the Y axis if angle is 0).

    angle : float
        Angle in degrees defining the ellipse rotation (clockwise tilt).

    Methods
    -------
    get_center()
        Returns center coordinates as tuple.

    point_on_ellipse(degrees)
        Returns point on ellipse perimeter.

    cv2_ellipse(start: float, end: float)
        Returns self as list, which when unpacked (star operator) is compatible with cv2.ellipse() drawing operation.
    """

    cx: int
    cy: int
    width: int
    height: int
    angle: float

    def __init__(self, cx: int, cy: int, width: int, height: int, angle: float):
        """
        cx : int
        Center X coordinate.

        cy : int
            Center Y coordinate.

        width : int
            Size of ellipse across the X axis.

        height : int
            Size of ellipse across the Y axis.

        angle : float
            Angle in degrees defining the ellipse rotation (clockwise tilt).
        """

        self.cx = int(cx)
        self.cy = int(cy)
        self.width = int(width)
        self.height = int(height)
        self.angle = angle

    def get_center(self) -> (int, int):
        """
        Returns center coordinates as tuple.

        Returns
        -------
        (int, int)
            X and Y coordinates.
        """

        return self.cx, self.cy

    def point_on_ellipse(self, degrees: float) -> (float, float):
        """
        Calculate point on ellipse perimeter, accounting for center offset.

        Parameters
        ----------
        degrees : float
            Angle on ellipse.

            0 and 180 lie on X axis (same X as center).

            90 and 270 lie on Y axis (same Y as center).

        Returns
        -------
        (float, float)
            X and Y coordinates of point on ellipse perimeter. X and Y coordinates in image (where ellipse is). Origin
            is in top left corner, NOT the ellipse center.
        """

        t = radians(degrees % 360)

        # Ellipse parameters
        theta = radians(self.angle)
        a = self.width / 2
        b = self.height / 2

        # Point on ellipse
        x = a * cos(t) * cos(theta) - b * sin(t) * sin(theta) + self.cx
        y = a * cos(t) * sin(theta) + b * sin(t) * cos(theta) + self.cy

        return x, y

    def cv2_ellipse(self, start: float = 0, end: float = 360) -> list:
        """
        Returns self as list, which when unpacked (star operator) is compatible with cv2.ellipse() drawing operation.
        For example as `cv2.ellipse(my_image, *my_ellipse.cv2_ellipse(), color=(255, 64, 255), thickness=5)`.

        Returns
        -------
        list
            List of (center coordinates), (axes), angle, start angle, end angle.
        """

        return [(self.cx, self.cy), (self.width // 2, self.height // 2), self.angle, start, end]


def scale_image(image: np.ndarray, factor: float, interpolation_method: int = None) -> np.ndarray:
    """
    Scale image by factor. Returns new instance of rescaled image.

    Parameters
    ----------
    image : np.ndarray
        Image to scale by factor.

    factor : float
        Multiplies width and height (resolution) of image.

    interpolation_method : int
        OpenCV interpolation method. If None, uses cv2.INTER_AREA when shrinking (factor < 1) and cv2.INTER_LINEAR when
        up-scaling (factor > 1). For more info about interpolation methods see OpenCV documentation
        (https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html).

        cv2.INTER_NEAREST : Nearest neighbor interpolation.

        cv2.INTER_LINEAR : Bilinear interpolation.

        cv2.INTER_CUBIC : Bicubic interpolation.

        cv2.INTER_AREA : Pixel area relation.

        cv2.INTER_LANCZOS4 : Lanczos interpolation over 8x8 neighborhood.

    Returns
    -------
    np.ndarray
        Scaled image.
    """

    if interpolation_method is None:
        if factor < 1:
            interpolation_method = cv2.INTER_AREA
        else:
            interpolation_method = cv2.INTER_LINEAR

    scaled_shape = (int(image.shape[1] * factor), int(image.shape[0] * factor))
    return cv2.resize(image, scaled_shape, interpolation_method)


def subsample_hash(array: np.ndarray, sample_size: int = 1024, seed: int = 666) -> int:
    """
    Hash of NumPy array using array samples. It is best to always use the same sample_size and seed, to avoid the
    possibility of the same array having different hashes.

    Parameters
    ----------
    array : np.ndarray
        NumPy array for which to compute hash.

    sample_size : int
        Number of samples to take. Higher number means more entropy, but slower computation. Using different sample
        sizes over the same array will most likely produce different hashes.

    seed : int
        Seed used to randomly choose samples. Using different seeds over the same array will most likely produce
        different hashes.

    Returns
    -------
    int
        Hash of taken samples.

    Notes
    -----
    Idea taken from https://stackoverflow.com/a/23300771.
    """

    # Choose random samples
    rng = np.random.RandomState(seed)
    indexes = rng.randint(low=0, high=array.size, size=sample_size)
    # Create new array of chosen samples
    sample = array.flat[indexes]
    sample.flags.writeable = False
    # Calculate the hash of the samples
    return hash(sample.data.tobytes())


def image_histogram(image: np.ndarray) -> np.ndarray:
    """
    Calculate normalized histogram of grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale, 2D array).

    Raises
    ------
    ValueError
        When input image isn't grayscale.

    Returns
    -------
    np.ndarray
        Normalized histogram (sum equals 1).
    """

    if len(image.shape) != 2:
        raise ValueError('Image is not grayscale.')

    histogram = cv2.calcHist(image, [0], None, [256], [0, 256], accumulate=False)
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return histogram
