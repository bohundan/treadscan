"""
This module is used for image segmentation, in particular to find the ellipse defining the tire position in image.

Assuming that the captured vehicle has bright enough wheel rim(s), as dark ones will not be contrasting with the tire
and will be lost during thresholding. Otherwise, the next step is to determine the tire's position by finding the
ellipse parameters (center of ellipse, height, width and angle).

The parameters defining an ellipse are the "output" of this module.
"""

from math import sin, cos, radians
from typing import Union

import cv2
import improutils
import numpy as np

from .utilities import Ellipse


class Segmentor:
    """
    Contains methods for image segmentation and ellipse detection.

    Attributes
    ----------
    image : numpy.ndarray
        Grayscale image on which to perform processing operations.

    Methods
    -------
    to_binary(threshold: int)
        Uses thresholding to create a binary image.

    filter_contours(binary_image: numpy.ndarray, min_area: int)
        Removes small and wide contours from binary image.

    find_ellipse(threshold: int, min_area: int)
        Finds and returns ellipse (car wheel/rim 'inside' tire).

    label_contours(threshold: int, min_area: int)
        Creates color image with labeled contours (contour area and score of fitted ellipse of each valid contour).
    """

    def __init__(self, image: np.ndarray):
        """
        Parameters
        ----------
        image : numpy.ndarray
            Grayscale image on which processing operations will be performed.

        Raises
        ------
        ValueError
            If image has invalid resolution.
        """

        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError('Image is zero pixels tall/wide.')

        if len(image.shape) != 2:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image

    def to_binary(self, threshold: int = 135) -> np.ndarray:
        """
        Creates a binary image using thresholding and filtering methods.

        Parameters
        ----------
        threshold : int
            Pixels higher than threshold will turn white, pixels smaller than threshold will turn black.

        Returns
        -------
        numpy.ndarray
            Binary image, original image is preprocessed before thresholding.

        Raises
        ------
        ValueError
            If threshold is not between 0-255.

            If kernel size is even or not greater than 1.
        """

        if not 0 <= threshold <= 255:
            raise ValueError('Invalid threshold value (must be between 0 and 255).')

        h, w = self.image.shape

        # Kernel with odd size
        kernel_size = h // 100 - (1 if (h // 100) % 2 == 0 else 0)
        kernel_size = max(3, kernel_size)

        # Blur image
        image = cv2.medianBlur(self.image, kernel_size)

        # Thresholding
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Bucket fill image corners (car tire should always be somewhere in the middle, never in a corner)
        cv2.floodFill(image, None, seedPoint=(0, 0), newVal=0)
        cv2.floodFill(image, None, seedPoint=(w - 1, 0), newVal=0)
        cv2.floodFill(image, None, seedPoint=(0, h - 1), newVal=0)
        cv2.floodFill(image, None, seedPoint=(w - 1, h - 1), newVal=0)

        # Fill holes
        image = improutils.fill_holes(image, close=True, size=h // 50)

        # Erosion, dilatation to get rid of some noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)

        return image

    @staticmethod
    def filter_contours(binary_image: np.ndarray, min_area: int = 0) -> list:
        """
        Remove small and wide contours from binary image.

        Parameters
        ----------
        binary_image : numpy.ndarray.

        min_area : int
            Minimum contour size (area), any smaller contours will be removed.

            If 0, computes min_area as (image width * height) // 100.

        Returns
        -------
        list of numpy.ndarray
            List of filtered contours.
        """

        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if min_area == 0:
            h, w = binary_image.shape
            min_area = w * h // 100

        filtered_contours = []
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            # Filter wide and small contours
            if h > w and cv2.contourArea(contour) > min_area:
                filtered_contours.append(contour)

        return filtered_contours

    @staticmethod
    def fit_ellipse(contour: np.ndarray) -> (Ellipse, float):
        """
        Fits an ellipse to contour.

        Parameters
        ----------
        contour : numpy.ndarray
            Contour to which fit an ellipse to.

        Returns
        -------
        (treadscan.Ellipse, float)
            Ellipse is defined by center coordinates, size and rotation.

            Second value is the error (sum of squares, the smaller, the better).
        """

        (cx, cy), (w, h), a = cv2.fitEllipse(contour)
        a_rad = radians(a)

        error = 0
        for point in contour:
            pos_x = (point[0][0] - cx) * cos(-a_rad) - (point[0][1] - cy) * sin(-a_rad)
            pos_y = (point[0][0] - cx) * sin(-a_rad) + (point[0][1] - cy) * cos(-a_rad)
            # -0.25 comes from the equation for ellipse
            # 0.25 instead of 1 because of full width and height (0.5**2)
            # https://answers.opencv.org/question/20521/how-do-i-get-the-goodness-of-fit-for-the-result-of-fitellipse/
            error += abs((pos_x / w) ** 2 + (pos_y / h) ** 2 - 0.25)

        return Ellipse(cx, cy, w, h, a), error

    def find_ellipse(self, threshold: int = 135, min_area: int = 0) -> Union[Ellipse, None]:
        """
        Find an ellipse in image (car wheel/rim).

        Parameters
        ----------
        threshold : int
            Thresholding threshold.

        min_area : int
            Minimum area of ellipse (ellipse contour area).

            0 for automatic (image width * height // 100).

        Returns
        -------
        treadscan.Ellipse
            Ellipse defined by center coordinates, size and rotation (in degrees).

        None
            If no ellipse was found.
        """

        binary_image = self.to_binary(threshold=threshold)
        viable_contours = self.filter_contours(binary_image, min_area=min_area)

        if not viable_contours:
            return None

        # Evaluating which contour best fits an ellipse, smaller the score the better fit
        best_ellipse = None
        best_score = float('inf')

        for contour in viable_contours:
            ellipse, error = self.fit_ellipse(contour)

            # Heuristic ellipse rating
            drawn_contour = improutils.crop_contour(contour, binary_image)
            solidity = improutils.solidity(drawn_contour)
            # 1.1 to avoid multiplying by (almost) zero
            score = error * (1.1 - solidity)

            if score < best_score:
                best_score = score
                best_ellipse = ellipse

        return best_ellipse

    def label_contours(self, threshold: int = 135, min_area: int = 0) -> np.ndarray:
        """
        Label each contour with its aspect ratio, area size and valid contours with the ellipse score of corresponding
        fitted ellipse. Labeled contours are stored in BGR image (3D numpy array).

        Parameters
        ----------
        threshold : int
            Thresholding threshold.

        min_area : int
            Minimum area of ellipse (ellipse contour area).

            0 for automatic (image width * height // 100).

        Returns
        -------
        numpy.ndarray
            BGR image with labeled contours.

            Red contour = too small.

            Yellow contour = bad aspect ratio (height should be greater than width).

            Green contour = valid contour with fitted ellipse. Smaller ellipse score means better fit.
        """

        binary_image = self.to_binary(threshold=threshold)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if min_area == 0:
            h, w = binary_image.shape
            min_area = w * h // 100

        # Classify contours
        small_contours = []
        wide_contours = []
        tall_contours = []
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) <= min_area:
                small_contours.append(contour)
            elif w >= h:
                wide_contours.append(contour)
            else:
                tall_contours.append(contour)

        # Empty color image on which to draw contours
        image = np.zeros(shape=(binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        cyan = (255, 255, 0)

        def put_text(text: str, location: (int, int), color: (int, int, int)):
            """
            Write outlined text on image.

            Parameters
            ----------
            text : str
                Text to write on image.

            location : (int, int)
                X and Y coordinates on image.

            color : (int, int, int)
                BGR color.
            """

            cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(50, 50, 50), thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=1,
                        lineType=cv2.LINE_AA)

        # Label small contours with just area
        for contour in small_contours:
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.drawContours(image, [contour], 0, red, cv2.FILLED)
            put_text(f'area: {int(cv2.contourArea(contour))}', (x, y), red)

        # Label wide contours with area and aspect ratio
        for contour in wide_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.drawContours(image, [contour], 0, yellow, cv2.FILLED)
            put_text(f'area: {int(cv2.contourArea(contour))}', (x, y), yellow)
            put_text(f'aspect ratio: {w / h:.1f}', (x, y + 30), yellow)

        # Label tall contours with area, aspect ratio and fitted ellipse
        for contour in tall_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.drawContours(image, [contour], 0, green, cv2.FILLED)
            # Fit ellipse
            ellipse, error = self.fit_ellipse(contour)
            drawn_contour = improutils.crop_contour(contour, binary_image)
            solidity = improutils.solidity(drawn_contour)
            score = error * (1.1 - solidity)
            # Draw ellipse and text
            cv2.ellipse(image, *ellipse.cv2_ellipse(), color=cyan, thickness=5, lineType=cv2.LINE_AA)
            put_text(f'area: {int(cv2.contourArea(contour))}', (x, y), green)
            put_text(f'aspect ratio: {w / h:.1f}', (x, y + 30), green)
            put_text(f'ellipse score: {score:.2f}', (x, y + 60), green)

        return image
