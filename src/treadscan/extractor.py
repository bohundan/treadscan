"""
This module is used for tread extraction (unwrapping) given the image and the ellipse defining the vehicle's rim in
image.

TireModel class defines a tire in an image. It can be created using 5 keypoints or just one ellipse (which represents
the vehicle's rim, tire sidewall and width is then approximated from the rim diameter.)
"""

from typing import Union

from .utilities import *


class TireModel:
    """
    Model representing a tire. Consists of two ellipses - outer and inner.

    Attributes
    ----------
    image_size : (int, int)
        Height and width of image in which the TireModel is placed (required for perspective transformations).

    main_ellipse : treadscan.Ellipse
        Ellipse defined by the wheel rim.

    outer_ellipse : treadscan.Ellipse
        Ellipse defined by the outer tire perimeter.

    inner_ellipse : treadscan.Ellipse
        Ellipse defined by the inner tire perimeter.

    beta_angle : float
        Angle of tire (in degrees) relative to camera's point of view.

    tire_sidewall : int
        Height of tire sidewall (in pixels).

    tire_width : int
        Tire width (in pixels).

    Methods
    -------
    from_keypoints(t: (int, int), b: (int, int), r: (int, int), s: (int, int), w: (int, int))
        Initializes TireModel using 5 keypoints.

    from_main_ellipse(main_ellipse: treadscan.Ellipse, tire_sidewall: int, tire_width: int, left_oriented: bool)
        Initializes TireModel using main ellipse (which is defined by the wheel rim).

    draw(image: numpy.ndarray, color: Union[int, tuple], thickness: int, lineType: int)
        Draws TireModel on image.

    bounding_box()
        Returns top left and bottom right points of TireModel's bounding box.

    unwrap(image: numpy.ndarray, start: int, end: int):
        Unwraps tread into a rectangle from an image.
    """

    def __init__(self, image_size: (int, int)):
        """
        Parameters
        ----------
        image_size : (int, int)
            Height and width of image.
        """

        self.image_size = image_size
        self.main_ellipse = None
        self.outer_ellipse = None
        self.inner_ellipse = None
        self.beta_angle = None
        self.tire_sidewall = None
        self.tire_width = None

    def from_keypoints(self, t: (int, int), b: (int, int), r: (int, int), s: (int, int), w: (int, int)):
        """
        Initializes TireModel using 5 keypoints.

        Parameters
        ----------
        t : (int, int)
            Top of rim.

        b : (int, int)
            Bottom of rim.

        r : (int, int)
            Third point on the rim perimeter.

        s : (int, int)
            Point above `t`, where tire sidewall ends.

        w : (int, int)
            Point on the inner side of tire.
        """

        self.main_ellipse = ellipse_from_points(t, b, r)
        self.beta_angle = self._calculate_tire_angle()

        self.outer_ellipse = Ellipse(self.main_ellipse.cx, self.main_ellipse.cy, self.main_ellipse.width,
                                     self.main_ellipse.height, self.main_ellipse.angle)
        self.outer_ellipse.fit_to_intersect(s)

        if self.outer_ellipse.is_point_inside(w):
            self.tire_width = 0
        else:
            self.tire_width = int(self.outer_ellipse.horizontal_distance_between_point(w))

        # Inner ellipse is on the left side of outer ellipse if the point on the inner side is left of ellipse center.
        left_oriented = w[0] < self.main_ellipse.cx
        self.inner_ellipse = self._create_inner_ellipse(left_oriented)

    def from_main_ellipse(self, main_ellipse: Ellipse, tire_sidewall: int = 0, tire_width: int = 0,
                          left_oriented: bool = False):
        """
        Initializes TireModel using main ellipse (which is defined by the wheel rim).

        Parameters
        ----------
        main_ellipse : treadscan.Ellipse
            Ellipse defined by wheel rim.

        tire_sidewall : int
            Height of tire sidewall in pixels. If 0, then the sidewall height is approximated as 1/5 of the wheel
            diameter.

        tire_width : int
            Width of tire in pixels. If 0, then the tire width is approximated as 1/2 of the wheel diameter.

        left_oriented : bool
            Defines if the inner side is on the left rather than the right. (TireModel is defined by outer and inner
            ellipses, inner ellipse is on the 'inside' of the wheel well, while the outer ellipse is visible.)
        """

        self.main_ellipse = main_ellipse
        self.beta_angle = self._calculate_tire_angle()

        self.tire_sidewall = tire_sidewall
        if self.tire_sidewall == 0:
            self.tire_sidewall = self.main_ellipse.height // 5
        self.outer_ellipse = self._create_outer_ellipse()

        self.tire_width = tire_width
        if self.tire_width == 0:
            self.tire_width = self.main_ellipse.height // 2
        self.inner_ellipse = self._create_inner_ellipse(left_oriented)

    def _calculate_tire_angle(self) -> float:
        """
        Returns
        -------
        float
            Angle of rotation of the tire in degrees (relative to camera's point of view).

        Raises
        ------
        RuntimeError
            When necessary attributes haven't been set.
        """

        if self.main_ellipse is None:
            raise RuntimeError('Main ellipse has not been set, cannot calculate tire angle.')

        # Calculate angle of wheel from ratio of ellipse axes, clip between -1 and 1 for arccos (ellipse should always
        # be taller rather than wider)
        ratio = np.clip(self.main_ellipse.width / self.main_ellipse.height, -1, 1)
        alpha = np.degrees(np.arccos(ratio))
        return 90 - alpha

    def _create_outer_ellipse(self) -> Ellipse:
        """
        Returns
        -------
        treadscan.Ellipse
            Ellipse extended (by tire sidewall) from main ellipse (wheel rim) to match outer tire perimeter.

        Raises
        ------
        RuntimeError
            When necessary attributes haven't been set.
        """

        if self.main_ellipse is None or self.tire_sidewall is None:
            raise RuntimeError('Main ellipse or tire sidewall has not been set, cannot create outer ellipse.')

        # Extend by tire sidewall
        height = int(self.main_ellipse.height + self.tire_sidewall * 2)
        size_coefficient = height / self.main_ellipse.height
        width = int(self.main_ellipse.width * size_coefficient)
        # Center stays the same
        return Ellipse(self.main_ellipse.cx, self.main_ellipse.cy, width, height, self.main_ellipse.angle)

    def _create_inner_ellipse(self, left_oriented: bool) -> Ellipse:
        """
        Parameters
        ----------
        left_oriented : bool
            If true, inner ellipse will be on the left side of outer ellipse.

        Returns
        -------
        treadscan.Ellipse
            Ellipse defined by the inner tire perimeter.

        Raises
        ------
        RuntimeError
            When necessary attributes haven't been set.
        """

        if self.image_size is None or self.outer_ellipse is None or self.tire_width is None:
            raise RuntimeError('Image size or outer ellipse or tire width has not been set, cannot create inner '
                               'ellipse.')

        temp_ellipse = Ellipse(self.outer_ellipse.cx, self.outer_ellipse.cy, self.outer_ellipse.width,
                               self.outer_ellipse.height, angle=0)
        top = temp_ellipse.point_on_ellipse(deg=-90)
        right = temp_ellipse.point_on_ellipse(deg=0)
        # left = self.outer_ellipse.point_on_ellipse(deg=-180)
        bottom = temp_ellipse.point_on_ellipse(deg=90)

        relative_width = self.tire_width
        if left_oriented:
            relative_width *= - 1

        # Shifted points
        top = int(top[0]) + relative_width, int(top[1])
        right = int(right[0]) + relative_width, int(right[1])
        bottom = int(bottom[0]) + relative_width, int(bottom[1])

        # Perspective transform
        beta = self.beta_angle

        center_offset = self.image_size[1] // 2 - int(temp_ellipse.point_on_ellipse(deg=-90)[0])
        top = perspective_transform_y_axis(beta, (top[0] + center_offset, top[1]),
                                           (self.image_size[0], self.image_size[1]))
        top = top[0] - center_offset, top[1]
        center_offset = self.image_size[1] // 2 - int(temp_ellipse.point_on_ellipse(deg=0)[0])
        right = perspective_transform_y_axis(beta, (right[0] + center_offset, right[1]),
                                             (self.image_size[0], self.image_size[1]))
        right = right[0] - center_offset, right[1]
        center_offset = self.image_size[1] // 2 - int(temp_ellipse.point_on_ellipse(deg=90)[0])
        bottom = perspective_transform_y_axis(beta, (bottom[0] + center_offset, bottom[1]),
                                              (self.image_size[0], self.image_size[1]))
        bottom = bottom[0] - center_offset, bottom[1]

        # Create new ellipse from transformed points
        cx = (top[0] + bottom[0]) // 2
        cy = right[1]
        width = abs(right[0] - cx) * 2
        height = abs(bottom[1] - top[1])

        cx = temp_ellipse.cx + relative_width
        cx, cy = rotate_point((cx, cy), self.outer_ellipse.angle, self.main_ellipse.get_center())

        return Ellipse(cx, cy, width, height, angle=self.outer_ellipse.angle)

    def draw(self, image: np.ndarray, color: Union[int, tuple], thickness: int = 1,
             lineType: int = cv2.LINE_8) -> np.ndarray:
        """
        Draws TireModel on image.

        Parameters
        ----------
        image: numpy.ndarray
            Image on which to draw on.

        color: Union[int, tuple]
            Color of resulting drawing (grayscale color or BGR, RGB... depending on image color scheme)

        thickness: int
            Thickness of the drawn TireModel.

        lineType: int
            Type of line (cv2.FILLED, cv2.LINE_4, cv2.LINE8 or cv2.LINE_AA).

        Returns
        -------
        numpy.ndarray
            Original image with drawn TireModel.

        Raises
        ------
        RuntimeError
            When TireModel is not initialized.
        """

        if self.inner_ellipse is None or self.outer_ellipse is None:
            raise RuntimeError('TireModel is not initialized.')

        # Inner ellipse 'orientation' (start and end are points on ellipse - represented by degrees).
        # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
        # 'If startAngle is greater than endAngle, they are swapped.' - reason behind the weird values.
        if self.outer_ellipse.cx < self.inner_ellipse.cx:
            start = -90
            end = 90
        else:
            start = 90
            end = 270

        image = cv2.ellipse(image, *self.outer_ellipse.cv2_ellipse(), color=color, thickness=thickness,
                            lineType=lineType)
        image = cv2.ellipse(image, *self.inner_ellipse.cv2_ellipse(start, end), color=color, thickness=thickness,
                            lineType=lineType)

        return image

    def bounding_box(self) -> ((int, int), (int, int)):
        """
        Create bounding box of tire model.

        Returns
        -------
        ((int, int), (int, int))
            Top left and bottom right points.

        Raises
        ------
        RuntimeError
            If TireModel is not initialized.
        """

        if self.outer_ellipse is None or self.inner_ellipse is None:
            raise RuntimeError('TireModel is not initialized.')

        # Find bounding box
        if self.outer_ellipse.cx < self.inner_ellipse.cx:
            left_bbox = self.outer_ellipse.bounding_box()
            right_bbox = self.inner_ellipse.bounding_box()
        else:
            left_bbox = self.inner_ellipse.bounding_box()
            right_bbox = self.outer_ellipse.bounding_box()

        # Extend bounding box to entire tire
        top_left = left_bbox[0]
        bottom_right = right_bbox[1][0], left_bbox[1][1]

        # Restrict points to within image
        left = min(self.image_size[1] - 1, max(0, top_left[0]))
        top = min(self.image_size[0], max(0, top_left[1]))
        right = max(0, min(self.image_size[1] - 1, bottom_right[0]))
        bottom = max(0, min(self.image_size[0] - 1, bottom_right[1]))

        return (left, top), (right, bottom)

    def unwrap(self, image: np.ndarray, start: int = -10, end: int = 80) -> np.ndarray:
        """
        Unwrap tire tread as a rectangle.

        Parameters
        ----------
        image : numpy.ndarray
            Image from which to unwrap the tread from.

        start : int
            Beginning (top) of tire tread represented in degrees.

        end : int
            End (bottom) of tire tread represented in degrees.

        Returns
        -------
        numpy.ndarray
            Image containing unwrapped tread.

        Raises
        ------
        RuntimeError
            When TireModel is not initialized or has 0 width (or height).
        """

        if self.outer_ellipse is None or self.inner_ellipse is None:
            raise RuntimeError('TireModel is not initialized.')
        if self.tire_width == 0:
            raise RuntimeError('TireModel is 0 pixels wide.')

        if start < -90 or end > 90:
            raise ValueError('Invalid start or end position, cannot unwrap tread out of view.')
        if start >= end:
            raise ValueError('Start angle has to be less than the end angle.')

        # Determining tread width
        point_a = self.outer_ellipse.point_on_ellipse(deg=0)
        point_b = self.inner_ellipse.point_on_ellipse(deg=0)
        tread_width = int(np.ceil(euclidean_dist(point_a, point_b)))

        if tread_width == 0:
            raise RuntimeError('Tread width equals 0, cannot proceed.')

        # Determining tread height and angular step size (on ellipse) in such a way, that the biggest step
        # (at 0 degrees) is the same length as the horizontal step
        horizontal_step_size = 1
        degrees_step_size = np.degrees(np.arcsin(2 * horizontal_step_size / self.outer_ellipse.height))
        total_degrees = abs(start - end)
        tread_height = int(np.ceil(total_degrees / degrees_step_size))

        if tread_height == 0:
            raise RuntimeError('Tread height equals 0, cannot proceed.')

        # Empty image (tread)
        shape = (tread_height, tread_width)
        if len(image.shape) == 3:
            shape = (tread_height, tread_width, image.shape[2])
        tread = np.zeros(shape=shape, dtype=np.uint8)

        left_ellipse = self.outer_ellipse
        right_ellipse = self.inner_ellipse
        if left_ellipse.cx > right_ellipse.cx:
            left_ellipse = self.inner_ellipse
            right_ellipse = self.outer_ellipse
            start = 180 - start
            end = 180 - end

        y = 0
        for deg in np.linspace(start, end, tread_height, endpoint=False):
            # Line is created between A and B
            point_a = left_ellipse.point_on_ellipse(deg=deg)
            point_b = right_ellipse.point_on_ellipse(deg=deg)
            x_step = (point_b[0] - point_a[0]) / tread_width
            y_step = (point_b[1] - point_a[1]) / tread_width
            for x in range(tread_width):
                # Step over the created line from left to right, extract pixels
                point = point_a[0] + x * x_step, point_a[1] + x * y_step
                pixel = cv2.getRectSubPix(image, (1, 1), point)[0, 0]
                tread[y, x] = pixel
            y += 1

        return tread
