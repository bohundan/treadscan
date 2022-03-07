"""
This module is used for tread extraction (unwrapping) given the image and the ellipse defining the car wheel in image.

Extractor class uses an image and an ellipse in the image, found by `Segmentor`, which defines the car wheel/rim, to
extract (unwrap) the tire tread as a separate image. First it corrects the rotation and then, using perspective
transformations, it creates bounding ellipses around the car tire. Tread unwrapping is then done by "walking" along
and between the two ellipses.
"""

from enum import Enum
from math import sqrt, sin, cos, asin, radians, degrees, ceil, floor, acos
import multiprocessing
from typing import Union

import cv2
import numpy as np
from .utilities import Ellipse


class CameraPosition(Enum):
    """
    Enum class specifying position from which car tire (wheel) has been captured.

    As most countries drive on the right side of the road, the safest placement would be on the right side of the road
    also (on the sidewalk instead of the middle of the road). And thus, the position would be on the RIGHT. This is
    important in tread extraction because of where the tread is visible would be different if the image is mirrored,
    as it would be normally be on the right side of the ellipse defining a car wheel.

    So if the image of the car is taken from the left side, it will simply be mirrored before extracting the tread and
    the tread mirrored again afterwards to correct orientation.

    FRONT_RIGHT :
        Camera facing car from the front, on the car's right side, capturing the RIGHT FRONT tire.

        (Right side tire captured from the front).

    FRONT_LEFT :
        Camera facing car from the front, on the car's left side, capturing the LEFT FRONT tire.

        (Left side tire captured from the front).

    BACK_RIGHT :
        Camera is BEHIND the car, on the car's right side, capturing the RIGHT BACK tire.

        (Right side tire captured from the back).

    BACK_LEFT :
        Camera is BEHIND the car, on the car's left side, capturing the LEFT BACK tire.

        (Left side tire captured from the back).
    """

    FRONT_RIGHT = 1
    FRONT_LEFT = 2
    BACK_RIGHT = 2
    BACK_LEFT = 1


class Extractor:
    """
    Extrapolates tire tread location in image from ellipse of wheel. Contains method for unwrapping the tire tread as
    a new separate image.

    Attributes
    ----------
    image : numpy.ndarray
        Image from which to extract tread.

    ellipse : treadscan.Ellipse
        Ellipse defined by center, size and rotation, which describes car's tire (wheel/rim) position in image.

    Methods
    -------
    perspective_transform_y_axis(angle: float, x: int, y: int)
        Returns new coordinates (position of original in image rotated around center Y axis).

    get_tire_bounding_ellipses(tire_width: int, tire_sidewall: int, outer_extend: int)
        Returns two ellipses, between which the tire occurs in (mirrored) image.

    visualise_bounding_ellipses(tire_width: int, tire_sidewall: int, outer_extend: int, start: float, end: float,
                                tire_bounding_ellipses: Union[tuple, None])
        Returns image with drawn tire model created from bounding ellipses.

    extract_tread(final_width: int, tire_width: int, start: float, end: float,
                  tire_bounding_ellipses: Union[tuple, None], cores: int)
        Returns new image with extracted tire tread (unwrapped).
    """

    def __init__(self, image: np.ndarray, ellipse: Ellipse, position: CameraPosition = CameraPosition.FRONT_RIGHT):
        """
        Parameters
        ----------
        image : numpy.ndarray
            Original image from which to extract tire tread.

        ellipse : treadscan.Ellipse
            Ellipse defined by center, size and rotation, which describes car's tire (wheel/rim) position in image.
        """

        self.main_ellipse = ellipse
        self.image = image

        # First transformation: unifying tread position relative to ellipse (tread always on the right side)
        if position in [CameraPosition.FRONT_LEFT, CameraPosition.BACK_RIGHT]:
            # Flip image horizontally (around Y axis)
            self.image = cv2.flip(self.image, 1)
            # Correct ellipse position accordingly
            self.main_ellipse.cx = self.image.shape[1] - self.main_ellipse.cx

        # Second transformation: correcting rotation
        angle = self.main_ellipse.angle
        if angle >= 90:
            angle -= 180
        rotation_matrix = cv2.getRotationMatrix2D(self.main_ellipse.get_center(), angle, scale=1.0)
        self.image = cv2.warpAffine(self.image, rotation_matrix, self.image.shape[1::-1], flags=cv2.INTER_LANCZOS4)
        self.main_ellipse.angle = 0

        # Remember if image is flipped
        self.flipped = position in [CameraPosition.FRONT_LEFT, CameraPosition.BACK_RIGHT]

    def perspective_transform_y_axis(self, angle: float, x: int, y: int) -> (int, int):
        r"""
        Perspective rotation around Y axis (center of image), takes original X and Y coordinates, returns transformed.

        Parameters
        ----------
        angle : float
            Angle of rotation in degrees (rotation around Y axis).

        x : int
            Original X coordinate.

        y : int
            Original Y coordinate.

        Returns
        -------
        (int, int)
            Tuple of transformed X and Y coordinates (position after perspective transformation).

        Notes
        -----
        :math:`A1` is a projection matrix from 2D to 3D, :math:`RY` is a rotation matrix (around the Y axis),
        :math:`T` is the translation matrix and :math:`A2` is a projection matrix back from 3D to 2D. [1]_

        .. math::

            A1 = \begin{pmatrix}
                    1 & 0 & -\frac{w}{2} \\
                    0 & 1 & -\frac{h}{2} \\
                    0 & 0 &       1      \\
                    0 & 0 &       1
                 \end{pmatrix}

            RY = \begin{pmatrix}
                    \cos(\varphi) & 0 & -\sin(\varphi) & 0 \\
                        0      & 1 &      0      & 0 \\
                    \sin(\varphi) & 0 & \cos(\varphi)  & 0 \\
                        0      & 0 &      0      & 1
                 \end{pmatrix}

            T = \begin{pmatrix}
                    1 & 0 & 0 \\
                    0 & 1 & 0 \\
                    0 & 0 & f \\
                    0 & 0 & 1
                 \end{pmatrix}

            A2 = \begin{pmatrix}
                    f & 0 & \frac{w}{2} & 0 \\
                    0 & f & \frac{h}{2} & 0 \\
                    0 & 0 &      1      & 0
                 \end{pmatrix}

        :math:`f` is calculated as :math:`\sqrt{\texttt{width}^2 + \texttt{height}^2}`.

        :math:`\varphi` is the angle of rotation around the Y axis.

        The final transformation matrix is then given by

        .. math:: M = \Big( A2 \cdot \big( T \cdot ( R \cdot A1 ) \big) \Big).

        And the transformation of the points :math:`x` and :math:`y` is [2]_

        .. math::

            \texttt{dst}(x, y) =
            \left(
                \frac{M_{11}x + M_{12}y + M_{13}}{M_{31}x + M_{32}y + M_{33}},
                \frac{M_{21}x + M_{22}y + M_{23}}{M_{31}x + M_{32}y + M_{33}}
            \right).

        .. [1] M. Jepson, https://jepsonsblog.blogspot.com/2012/11/rotation-in-3d-using-opencvs.html
           28 November 2012
        .. [2] OpenCV documentation,
           https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
           3 February 2022
        """

        # Image height and width
        h, w = self.image.shape[0], self.image.shape[1]
        # Focal point
        f = sqrt(w**2 + h**2)

        phi = radians(angle)

        # Transformation matrix, explained in notes section
        m_11 = f * cos(phi) + (w / 2) * sin(phi)
        m_12 = 0
        m_13 = f * ((-w / 2) * cos(phi) - sin(phi)) + (w / 2) * ((-w / 2) * sin(phi) + cos(phi) + f)
        m_21 = (h / 2) * sin(phi)
        m_22 = f
        m_23 = f * (-h / 2) + (h / 2) * ((-w / 2) * sin(phi) + cos(phi) + f)
        m_31 = sin(phi)
        m_32 = 0
        m_33 = (-w / 2) * sin(phi) + cos(phi) + f

        # Transformed coordinates
        x_ = (m_11 * x + m_12 * y + m_13) / (m_31 * x + m_32 * y + m_33)
        y_ = (m_21 * x + m_22 * y + m_23) / (m_31 * x + m_32 * y + m_33)

        return x_, y_

    def get_tire_bounding_ellipses(self, tire_width: int = 0, tire_sidewall: int = 0, outer_extend: int = 0) \
            -> (Ellipse, Ellipse):
        """
        Create and return outer and inner ellipses surrounding car tire.

        These ellipses may be flipped the wrong way around (outer is always on the left and inner is always on the
        right) as there is no easy way to determine on which side of the ellipse is the tire tread visible,
        refer to `CameraPosition`.

        Parameters
        ----------
        tire_width : int
            Tire width in pixels, 0 for automatic (half of wheel diameter).

        tire_sidewall : int
            Height of tire sidewall in pixels, for automatic (half of tire width).

        outer_extend : int
            If non-zero, extend outer ellipse (outwards) by this much. To make sure that entirety of tire tread is
            visible between bounding ellipses.

        Returns
        -------
        (treadscan.Ellipse, treadscan.Ellipse)
            Tuple of outer and inner ellipses, each defined by center, height and width.
        """

        if tire_width < 0:
            raise ValueError('Tire width has to be greater than 0.')
        if tire_sidewall < 0:
            raise ValueError('Tire sidewall has to be greater than 0.')

        # Calculate angle of wheel from ratio of ellipse axes, clip between -1 and 1 for arccos (ellipse should always
        # be taller rather than wider anyway)
        ratio = np.clip(self.main_ellipse.width / self.main_ellipse.height, -1, 1)
        alpha = degrees(acos(ratio))
        beta = 90 - alpha
        # Example: 205/55 R 16, width is about half the wheel diameter, sidewall about quarter the wheel diameter
        #          205 mm tire width
        #          205 mm * 0.55 = 113 mm tire sidewall
        #          16inch = 406 mm wheel diameter
        if tire_width == 0:
            tire_width = int(self.main_ellipse.height / 2)
        if tire_sidewall == 0:
            # After some experimentation, it seems 1/5 is a better factor than 1/4
            tire_sidewall = int(self.main_ellipse.height / 5)

        # Extend main ellipse to match tire perimeter
        height2 = self.main_ellipse.height + tire_sidewall * 2
        size_coefficient = height2 / self.main_ellipse.height
        width2 = int(self.main_ellipse.width * size_coefficient)
        # Center stays the same
        tire_ellipse = Ellipse(self.main_ellipse.cx, self.main_ellipse.cy, width2, height2, angle=0)

        def perspective_shift_ellipse(ellipse: Ellipse, shift_by: int) -> Ellipse:
            """
            Apply perspective shift to ellipse.

            Parameters
            ----------
            ellipse : treadscan.Ellipse
                Ellipse which to transform.

            shift_by : int
                How far to move ellipse (negative number to move left, positive to move right) on shifted X axis.
                (Along perspective rotated X axis - rotated around center Y axis).

            Returns
            -------
            treadscan.Ellipse
                New ellipse moved across the (rotated) X axis.
            """

            # Shift these 3 points, then use them to recreate ellipse
            top = ellipse.point_on_ellipse(deg=-90)
            right = ellipse.point_on_ellipse(deg=0)
            # left = ellipse.point_on_ellipse(deg=-180)
            bottom = ellipse.point_on_ellipse(deg=90)

            # Shifted points
            top = int(top[0]) + shift_by, int(top[1])
            right = int(right[0]) + shift_by, int(right[1])
            bottom = int(bottom[0]) + shift_by, int(bottom[1])

            # Perspective transform
            center_offset = self.image.shape[1] // 2 - int(ellipse.point_on_ellipse(deg=-90)[0])
            top = self.perspective_transform_y_axis(beta, top[0] + center_offset, top[1])
            top = top[0] - center_offset, top[1]
            center_offset = self.image.shape[1] // 2 - int(ellipse.point_on_ellipse(deg=0)[0])
            right = self.perspective_transform_y_axis(beta, right[0] + center_offset, right[1])
            right = right[0] - center_offset, right[1]
            center_offset = self.image.shape[1] // 2 - int(ellipse.point_on_ellipse(deg=90)[0])
            bottom = self.perspective_transform_y_axis(beta, bottom[0] + center_offset, bottom[1])
            bottom = bottom[0] - center_offset, bottom[1]

            # Create new ellipse
            cx = (top[0] + bottom[0]) // 2
            cy = right[1]
            width = abs(right[0] - cx) * 2
            height = bottom[1] - top[1]
            shifted_ellipse = Ellipse(cx, cy, width, height, angle=0)

            return shifted_ellipse

        # Shift inner ellipse by tire_width
        inner_ellipse = perspective_shift_ellipse(tire_ellipse, tire_width)

        # Shift outer ellipse (in the opposite direction) by outer_extend
        outer_ellipse = perspective_shift_ellipse(tire_ellipse, -outer_extend)

        return outer_ellipse, inner_ellipse

    def visualise_bounding_ellipses(self, tire_width: int = 0, tire_sidewall: int = 0, outer_extend: int = 0,
                                    start: float = -10, end: float = 80,
                                    tire_bounding_ellipses: Union[tuple, None] = None) -> np.ndarray:
        """
        Creates color (BGR) image with drawn tire model.

        Parameters
        ----------
        tire_width : int
            Width of tire in original image (in pixels). Is only used when auto-generating tire bounding ellipses.

            0 for automatic (overestimates on purpose to avoid missing tread).

        tire_sidewall : int
            Height of tire sidewall in pixels, for automatic (half of tire width).

            0 for automatic (quarter of wheel diameter).

        outer_extend : int
            If non-zero, extend outer ellipse (outwards) by this much. To make sure that entirety of tire tread is
            visible between bounding ellipses.

        start : float
            Starting angle of extraction (must be between -90 and 90, -90 is top of tire, 0 is middle, 90 is bottom).

        end : float
            End angle of extraction (must be greater than `start` and must be between -90 and 90).

        tire_bounding_ellipses : Union[tuple, None]
            Outer and inner tire bounding ellipses.

            If none, they will be auto-generated.

        Returns
        -------
        numpy.ndarray
            BGR image with drawn ellipses defining the tire model used for tread extraction.

            Red ellipse = main ellipse (car wheel).

            Yellow ellipses = bounding ellipses of tire (outer and inner sides of tire).

            Green rectangle = area of tread extraction.

        Raises
        ------
        ValueError
            When any of parameters are invalid (negative resolution, wrong degrees, invalid ellipses etc.).
        """

        if start < -90 or end > 90:
            raise ValueError('Invalid start or end position, cannot extract tread out of view.')
        if start >= end:
            raise ValueError('Start angle has to be less than the end angle.')
        if tire_width < 0:
            raise ValueError('Tire width must be greater than 0.')
        if tire_bounding_ellipses is not None:
            if len(tire_bounding_ellipses) != 2:
                raise ValueError('You must provide exactly 2 ellipses.')
            if not isinstance(tire_bounding_ellipses[0], Ellipse) or not isinstance(tire_bounding_ellipses[1], Ellipse):
                raise ValueError('One of the provided ellipses is not an instance of treadscan.Ellipse.')

        if tire_bounding_ellipses is None:
            tire_bounding_ellipses = self.get_tire_bounding_ellipses(tire_width, tire_sidewall, outer_extend)

        outer, inner = tire_bounding_ellipses

        if outer.cx > inner.cx:
            raise ValueError('Ellipses are crossed, outer should be on the left side, inner on the right side.')

        start_a = outer.point_on_ellipse(start)
        start_b = inner.point_on_ellipse(start)
        end_a = outer.point_on_ellipse(end)
        end_b = inner.point_on_ellipse(end)

        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        yellow = (0, 255, 255)
        green = (0, 255, 0)

        # Draw horizontal lines, top and bottom of tread
        cv2.line(image, (int(start_a[0]), int(start_a[1])), (int(start_b[0]), int(start_b[1])), thickness=5,
                 color=green)
        cv2.line(image, (int(end_a[0]), int(end_a[1])), (int(end_b[0]), int(end_b[1])), thickness=5, color=green)

        # Draw bounding ellipses
        cv2.ellipse(image, *outer.cv2_ellipse(), thickness=5, color=yellow)
        cv2.ellipse(image, *inner.cv2_ellipse(-90, 90), thickness=5, color=yellow)
        # Edges of tire tread
        cv2.ellipse(image, *outer.cv2_ellipse(start, end), thickness=5, color=green)
        cv2.ellipse(image, *inner.cv2_ellipse(start, end), thickness=5, color=green)

        return image

    def extract_tread(self, final_width: int = 0, tire_width: int = 0, start: float = -10, end: float = 80,
                      tire_bounding_ellipses: Union[tuple, None] = None, cores: int = 4) -> np.ndarray:
        """
        Unwraps tire tread into a new image.

        Parameters
        ----------
        final_width : int
            Width of final image (unwrapped tread). This essentially scales the final image to desired width, height
            stays proportional.

            0 for automatic (same size as tire_width).

        tire_width : int
            Width of tire in original image (in pixels). Is only used when auto-generating tire bounding ellipses.

            0 for automatic (overestimates on purpose to avoid missing tread).

        start : float
            Starting angle of extraction (must be between -90 and 90, -90 is top of tire, 0 is middle, 90 is bottom).

        end : float
            End angle of extraction (must be greater than `start` and must be between -90 and 90).

        tire_bounding_ellipses : Union[tuple, None]
            Outer and inner tire bounding ellipses.

            If none, they will be auto-generated.

        cores : int
            Number of child processes to spawn for parallel tread extraction.

            Be warned that each spawned child process creates a copy of the original image, this can eat up a lot of
            memory if hundreds are spawned. This doesn't happen when forking processes (copy-on-write), fork() is only
            available on POSIX-compliant systems.

        Returns
        -------
        numpy.ndarray
            New image with extracted tread. If original image has 1 channel (grayscale), output of this function is
            also a grayscale image. If original has more channels (BGR, RGB, etc.), output is also in the same
            colorspace (BGR, RGB, etc.).

        Raises
        ------
        ValueError
            When any of parameters are invalid (negative resolution, wrong degrees, invalid ellipses etc.).
        """

        if start < -90 or end > 90:
            raise ValueError('Invalid start or end position, cannot extract tread out of view.')
        if start >= end:
            raise ValueError('Start angle has to be less than the end angle.')
        if final_width < 0:
            raise ValueError('Final width must be greater than 0.')
        if tire_width < 0:
            raise ValueError('Tire width must be greater than 0.')
        if cores < 1:
            raise ValueError('Number of cores must be greater or equal to 1.')

        if tire_width == 0:
            tire_width = int(self.main_ellipse.height / 1.8)

        tread_width = final_width
        if tread_width == 0:
            tread_width = tire_width

        if tire_bounding_ellipses is None:
            outer_ellipse, inner_ellipse = self.get_tire_bounding_ellipses(tire_width=tire_width)
        else:
            if len(tire_bounding_ellipses) != 2:
                raise ValueError('Invalid amount of ellipses provided. Must be 2 (outer and inner).')
            outer_ellipse, inner_ellipse = tire_bounding_ellipses
            if type(outer_ellipse) is not Ellipse or type(inner_ellipse) is not Ellipse:
                raise ValueError('Ellipses are not an instance of treadscan.Ellipse.')
            # Rotation is already accounted for
            outer_ellipse.angle = 0
            inner_ellipse.angle = 0

        # Determining horizontal step size
        point_a = outer_ellipse.point_on_ellipse(deg=0)
        point_b = inner_ellipse.point_on_ellipse(deg=0)
        # Euclidean distance between points
        tire_width_in_image = sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
        step_size = tire_width_in_image / tread_width

        # Determining angular step size (on ellipse) in such a way, that the biggest step (at 0 degrees) is the same
        # length as the horizontal steps
        degrees_step_size = degrees(asin(2 * step_size / outer_ellipse.height))
        total_degrees = abs(start - end)
        tread_height = ceil(total_degrees / degrees_step_size)

        # Parallel tread unwrapping, each core (process) is assigned one part of tire tread, which is split horizontally
        part = floor(tread_height / cores)
        # Each part is the same height, except possibly the last one, which can be taller by upto (cores - 1) pixels
        remainder = int(tread_height / cores % 1 * cores)
        # This list contains first and last Y coordinates, for example 40 pixels tall tread, then [0, 10, 20, 30, 40]
        parts = [part * i for i in range(0, cores + 1)]
        # Don't forget the remainder
        parts[-1] = parts[-1] + remainder

        # Sanity check
        if len(parts) > 1 and parts[1] - parts[0] < 1:
            raise ValueError('Tread split into too many parts (reduce number of cores).')

        def collect_args(index: int) -> (int, np.ndarray):
            """
            Collects required parameters for one part of tread.

            Returns
            -------
            (int, numpy.ndarray)
                Tuple of index of part and degree range.
            """

            range_start = start + parts[index] * degrees_step_size
            range_stop = start + parts[index + 1] * degrees_step_size
            height = parts[index + 1] - parts[index]
            # Exclude endpoint for same behavior as np.arange, np.arange can be unstable, therefore using np.linspace
            linspace = np.linspace(range_start, range_stop, height, endpoint=False)

            return index, linspace

        # Write variables, which are shared across all child processes, into shared memory (to avoid copying when
        # using fork() to create processes)
        global global_image
        global global_outer_ellipse
        global global_inner_ellipse
        global global_tread_width
        global_image = self.image
        global_outer_ellipse = outer_ellipse
        global_inner_ellipse = inner_ellipse
        global_tread_width = tread_width

        # Unwrap tread in parallel
        pool = multiprocessing.Pool(processes=cores)
        # List of tuples, each process returns index of part and part itself (image)
        result = pool.starmap(tire_tread_part, [collect_args(i) for i in range(0, cores)])

        # Sort and stitch parts together (first element in tuple is index, second the tread part image)
        sorted(result, key=lambda x: x[0])
        sorted_parts = [r[1] for r in result]
        tread = np.concatenate(sorted_parts)

        # Clean up global variables to avoid any misuse
        global_image = None
        global_outer_ellipse = None
        global_inner_ellipse = None
        global_tread_width = None

        if self.flipped:
            tread = cv2.flip(tread, 1)
        return tread


global_image = None
global_outer_ellipse = None
global_inner_ellipse = None
global_tread_width = None


def tire_tread_part(index: int, linspace: np.ndarray) -> (int, np.ndarray):
    """
    Unwraps part of tire tread.

    Parameters
    ----------
    index : int
        Index of part (to identify where part belongs).

    linspace : numpy.ndarray
        Range of degrees on bounding (outer and inner) ellipses, which is being unwrapped.

    Returns
    -------
    (int, numpy.ndarray)
        Index of tread part and unwrapped tread as image.

    Raises
    ------
    RuntimeError
        When variables in shared memory haven't been set.

    Notes
    -----
    Uses variables in shared memory. Without setting these first, the method does nothing. Shared memory is used to gain
    significant performance advantages over copying the variables for each process, since this method is used for
    parallel extraction of tire tread. Copying the original image (and other arguments) is slow, especially if hundreds
    of child processes are spawned, which is admittedly an edge case, but consider a super huge mega large image,
    copying it many times might even be impossible (not enough memory) so the extraction would need to be run on
    a single core only to avoid OOM, in this case, the ability to use possibly hundreds of cores to speed up the
    extraction offers great performance gain for almost no downsides. (The real downside is spaghetti-like code using
    global variables and this method being outside the Extractor class, which is again, to avoid unnecessary copying).

    This performance gain exists only on POSIX-compliant systems, because the fork() child "points" to the same memory
    as the parent and uses copy-on-write strategy. Since this method only reads from the original image and doesn't
    modify it, new copies don't need to be made, saving time and memory.
    That doesn't happen using the win32 API, which creates an entirely new process and all the data needs to be copied
    because of that (https://stackoverflow.com/a/17786444).
    """

    if global_image is None or global_outer_ellipse is None or global_inner_ellipse is None \
            or global_tread_width is None:
        raise RuntimeError('Global variables have not been set, cannot proceed.')

    # Create empty image
    tread_height = len(linspace)
    if len(global_image.shape) > 2:
        # Color
        tread_part = np.zeros(shape=(tread_height, global_tread_width, global_image.shape[2]), dtype=np.uint8)
    else:
        # Grayscale
        tread_part = np.zeros(shape=(tread_height, global_tread_width), dtype=np.uint8)

    y = 0
    for deg in linspace:
        # Point A lies on outer ellipse, point B on inner ellipse
        # Line is created between A and B
        point_a = global_outer_ellipse.point_on_ellipse(deg=deg)
        point_b = global_inner_ellipse.point_on_ellipse(deg=deg)
        x_step = (point_b[0] - point_a[0]) / global_tread_width
        y_step = (point_b[1] - point_a[1]) / global_tread_width
        for x in range(global_tread_width):
            # Step over the created line from left to right, extract pixels
            point = point_a[0] + x * x_step, point_a[1] + x * y_step
            pixel = cv2.getRectSubPix(global_image, (1, 1), point)[0, 0]
            tread_part[y, x] = pixel
        y += 1

    return index, tread_part
