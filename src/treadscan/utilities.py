"""
This module contains various useful methods that can be used in multiple different places.
"""

from os.path import isfile
from typing import Union

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

    horizontal_distance_between_point(x: int, y: int)
        Returns the length of a horizontal line drawn from point and intersecting with the ellipse. Only implemented for
        non-rotated ellipse.

    bounding_box():
        Returns top left and bottom right points of ellipse's bounding box.

    area():
        Returns ellipse's area.
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

    def __str__(self):
        """
        Returns string describing ellipse.
        """

        return f'Center: {self.cx}, {self.cy}\nAngle: {self.angle}\nHeight: {self.height}\nWidth: {self.width}'

    def get_center(self) -> (int, int):
        """
        Returns center coordinates as tuple.

        Returns
        -------
        (int, int)
            X and Y coordinates.
        """

        return self.cx, self.cy

    def point_on_ellipse(self, deg: float) -> (float, float):
        """
        Calculate point on ellipse perimeter, accounting for center offset.

        Parameters
        ----------
        deg : float
            Angle on ellipse.

            0 and 180 lie on X axis (same X as center).

            90 and 270 lie on Y axis (same Y as center).

        Returns
        -------
        (float, float)
            X and Y coordinates of point on ellipse perimeter. X and Y coordinates in image (where ellipse is). Origin
            is in top left corner, NOT the ellipse center.
        """

        t = np.radians(deg % 360)

        # Ellipse parameters
        theta = np.radians(self.angle)
        a = self.width / 2
        b = self.height / 2

        # Point on ellipse
        x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta) + self.cx
        y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta) + self.cy

        return x, y

    def horizontal_distance_between_point(self, x: int, y: int) -> int:
        """
        Approximate angle which gives the closest point on the right side of ellipse to provided point.
        Only works for ellipses which AREN'T ROTATED.

        Parameters
        ----------
        x : int
            X coordinate of point.
        y : int
            Y coordinate of point.

        Returns
        -------
        int
            Length of horizontal line drawn from (x, y) and intersecting ellipse.

        Raises
        ------
        RuntimeError
            When ran for a rotated ellipse (ellipse.angle != 0) or when ellipse has 0 width.
        """

        if self.angle != 0:
            raise RuntimeError('This method does not work correctly with a rotated ellipse.')
        if self.width == 0:
            raise RuntimeError('Ellipse has 0 width.')

        a = self.width / 2
        b = self.height / 2

        # Move Y to count for ellipse being off center (not at 0,0)
        y -= self.cy
        # Make sure Y is on ellipse
        y = np.clip(y, -b + 2, b - 2)  # Magic two to make really sure

        # From the equation for ellipse x^2/a^2 + y^2/b^2 = 1, x = a * sqrt(b**2 - y**2) / b
        # Avoid division by zero (and square root of negative number)
        if b <= abs(y):
            b = y + 5  # Magic five
        x_on_ellipse = a * np.sqrt(b**2 - y**2) / b

        # Correct side
        if x < self.cx:
            x_on_ellipse *= -1

        dist = abs(int(self.cx + x_on_ellipse - x))

        return dist

    def bounding_box(self) -> list:
        """
        Create bounding box around ellipse. Compatible with OpenCV's cv2.rectangle() drawing operation (returns points
        as integer tuples).

        Returns
        -------
        list
            List of two points (tuples) - top left and bottom right corner of bounding box.
        """

        theta = np.radians(self.angle)

        a = self.height / 2
        b = self.width / 2

        # Ellipse limits on given axis
        y = np.sqrt(a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2)
        x = np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2)

        top_left = self.cx - int(x), self.cy - int(y)
        bottom_right = self.cx + int(x), self.cy + int(y)

        return [top_left, bottom_right]

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

    def area(self) -> float:
        """
        Calculate area of ellipse.

        Returns
        -------
        float
            Area of ellipse.
        """

        return np.pi * (self.height / 2) * (self.width / 2)


def ellipse_from_points(top: (int, int), bottom: (int, int), third: (int, int)) -> Ellipse:
    """
    Create ellipse from 3 points. Top and bottom are ellipse vertices, third point lies anywhere on the ellipse.

    Parameters
    ----------
    top : (int, int)
        X and Y coordinates of the top of the ellipse (at -90 degrees).

    bottom : (int, int)
        X and Y coordinates of the bottom of the ellipse (at 90 degrees).

    third : (int, int)
        X and Y coordinates of any point on the ellipse (except top or bottom).

    Returns
    -------
    treadscan.Ellipse
        Ellipse constructed from the provided points.
    """

    # Center is between top and bottom, in the middle
    cx = (bottom[0] + top[0]) / 2
    cy = (bottom[1] + top[1]) / 2

    # Height is the Euclidean distance between top and bottom
    height = euclidean_dist(top, bottom)

    # Calculating angle using a vector drawn between center and top of ellipse
    x = abs(cx - top[0])
    y = cy - top[1]

    # Avoid division by zero
    if x == 0:
        angle = 0
    else:
        angle = 90 - np.degrees(np.arctan(y / x))
    # Flip angle to the other side if ellipse is leaning to the left
    angle *= -1 if top[0] < cx else 1

    # Third point might not be exactly the right vertex on ellipse (at 0 degrees)
    # Instead use point coordinates to solve ellipse equation
    # First rotate point the other way for simpler equation (removes the ugly trigonometry)
    p = rotate_point(third, -angle, (int(cx), int(cy)))

    # Ellipse equation is x^2/a^2 + y^2/b^2 = 1, which means a = b * x / sqrt(b^2 - y^2)
    b = height / 2
    # Avoid division by zero (and square root of negative number)
    if b <= abs(p[1] - cy):
        b = p[1] - cy + 5  # Magic five
    a = b * (p[0] - cx) / np.sqrt(b**2 - (p[1] - cy)**2)

    width = abs(2 * a)

    if np.isnan(width):
        width = 0

    return Ellipse(int(cx), int(cy), int(width), int(height), angle)


def euclidean_dist(a: Union[tuple, list], b: Union[tuple, list]) -> float:
    """
    Return the Euclidean distance between two points

    Parameters
    ----------
    a: Union[(int, int), [int, int]]

    b: Union[(int, int), [int, int]]

    Returns
    -------
    float
        Euclidean distance between points
    """

    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def rotate_point(point: (int, int), angle: float, pivot: (int, int) = (0, 0)) -> (int, int):
    """
    Return point rotated by angle around given origin.

    Parameters
    ----------
    point : (int, int)
        Original point coordinates.

    angle : float
        Angle in degrees to rotate point by.

    pivot: (int, int)
        Center of rotation, (0, 0) by default.

    Returns
    -------
    (int, int)
        New coordinates of point.

    Notes
    -----
    Source: https://stackoverflow.com/a/15109215.
    """

    angle = np.radians(angle)

    x_ = np.cos(angle) * (point[0] - pivot[0]) - np.sin(angle) * (point[1] - pivot[1]) + pivot[0]
    y_ = np.sin(angle) * (point[0] - pivot[0]) + np.cos(angle) * (point[1] - pivot[1]) + pivot[1]

    return int(x_), int(y_)


def load_image(path: str):
    """
    Load grayscale image from given path.

    Parameters
    ----------
    path : str
        Path to image.

    Returns
    -------
    numpy.ndarray
        Grayscale image as 2D array.

    Raises
    ------
    ValueError
        When file does not exist or is not readable.
    """

    if isfile(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('File does not exists or is not readable.')

    return image


def scale_image(image: np.ndarray, factor: float, interpolation_method: int = None) -> np.ndarray:
    """
    Scale image by factor. Returns new instance of rescaled image.

    Parameters
    ----------
    image : numpy.ndarray
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
    numpy.ndarray
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
    array : numpy.ndarray
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
    image : numpy.ndarray
        Input image (grayscale, 2D array).

    Raises
    ------
    ValueError
        When input image isn't grayscale.

    Returns
    -------
    numpy.ndarray
        Normalized histogram (sum equals 1).
    """

    if len(image.shape) != 2:
        raise ValueError('Image is not grayscale.')

    histogram = cv2.calcHist(image, [0], None, [256], [0, 256], accumulate=False)
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return histogram
