"""
This module contains the Annotator class which can be used to obtain ellipse which defines the car wheel (and tire),
for example if treadscan Segmentor fails (like when wheel is too dark) to detect the main ellipse.
"""

from datetime import datetime
from math import sqrt, asin, degrees, atan, radians
import json
from typing import Union
import sys

import cv2
import numpy as np
from treadscan.extractor import Extractor, CameraPosition
from treadscan.utilities import Ellipse, ellipse_from_points, scale_image


class Annotator:
    """
    Class containing methods for manual creation of bounding ellipses that define tire tread in image, using cv2 GUI
    methods.

    Attributes
    ----------
    image : numpy.ndarray
        Image being annotated.

    scale : float
        Scale of shown image compared to original.

    flipped : bool
        Whether image has been flipped horizontally or not.

    mouse_pos : (int, int)
        Position of mouse over image (X and Y coordinates).

    points : dict[int, Any]
        Contains position of all the annotation points.
    """

    def __init__(self, image: np.ndarray, max_width: int, max_height: int, flipped: bool = False):
        """
        Parameters
        ----------
        image : np.ndarray
            Original image which will be annotated.

        max_width : int
            Maximum width of annotation window (without preview).

        max_height : int
            Maximum height of annotation window (without preview).

        flipped : bool
            If true, image will be flipped horizontally beforehand.
        """

        self.scale = min(max_width / image.shape[1], max_height / image.shape[0])
        self.scale = min(self.scale, 1)

        self.image = scale_image(image, self.scale)
        self.flipped = flipped
        if self.flipped:
            self.image = cv2.flip(self.image, 1)
        # Convert grayscale to BGR
        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            self.image = image.copy()

        self.mouse_pos = None
        self.points = {
            ord('t'): None,  # Top
            ord('b'): None,  # Bottom
            ord('r'): None,  # Right
            ord('s'): None,  # Tire sidewall
            ord('o'): None,  # Outer extend
            ord('w'): None,  # Tire width
            ord('u'): None,  # Start angle
            ord('i'): None   # End angle
        }

        # Hidden attributes defining some default values
        self._default_start_angle = -10
        self._default_end_angle = 80
        self._default_tire_width_from_wheel_diameter_coefficient = 1.8
        # Really hidden attributes for less wonky behaviour of some annotation points to stop movement of multiple
        # points as a reaction to one point
        self.__prev_start = None
        self.__prev_end = None
        self.__prev_top = (0, 0)
        self.__prev_bottom = (0, 0)

    def draw_only_annotation_points(self, image: np.ndarray):
        """
        Draw each point labeled with the key used to set it.

        Parameters
        ----------
        image : numpy.ndarray
            Image on which to draw annotation points.
        """

        point_size = 4

        # Unused points are drawn in top left corner in one column
        y_pos = 8
        for key, value in self.points.items():
            if value is not None:
                # Point is set, draw it where it is set
                cv2.circle(image, value, point_size, (0, 128, 255), -1)
                point = (value[0] - 4, value[1] + 4)
                cv2.putText(image, chr(key), point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0),
                            thickness=1)
            else:
                # Point is not set, draw it at the bottom of the column
                cv2.circle(image, (4, y_pos), point_size, (0, 128, 255), -1)
                cv2.putText(image, chr(key), (0, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(0, 0, 0), thickness=1)
                y_pos += 16

    def draw_points(self, image: np.ndarray) -> Union[None, tuple]:
        """
        Draws points over image, builds ellipse parameters and bounding ellipses.

        Parameters
        ----------
        image : numpy.ndarray
            Image over which to draw, should be in BGR.

        Returns
        -------
        None
            If no ellipses can be constructed.

        (treadscan.Extractor, treadscan.Ellipse, treadscan.Ellipse, treadscan.Ellipse, int, int, float, float)
            Tuple of extractor, main ellipse, outer ellipse, inner ellipse, tire width, outer ellipse extend,
            start and end angles.
            When ellipses CAN be constructed from points annotated on image.
        """

        top = self.points[ord('t')]
        bottom = self.points[ord('b')]
        right = self.points[ord('r')]
        sidewall = self.points[ord('s')]
        outer_extend = self.points[ord('o')]
        width = self.points[ord('w')]
        start = self.points[ord('u')]
        end = self.points[ord('i')]

        # Calculate tire sidewall height as distance between top or bottom point and the sidewall point
        if sidewall and top and bottom:
            cx = abs(bottom[1] + top[1]) // 2
            if sidewall[1] < cx:
                sidewall = max(1, top[1] - sidewall[1])
            else:
                sidewall = max(1, sidewall[1] - bottom[1])
        else:
            sidewall = 0

        # Calculate tire width as distance from center of main ellipse
        if width and top and bottom:
            width = max(10, width[0] - (top[0] + bottom[0]) // 2)
        else:
            width = 0

        # Calculate extension of outer ellipse as distance from center of main ellipse
        if outer_extend and top and bottom:
            outer_extend = (top[0] + bottom[0]) // 2 - outer_extend[0]
        else:
            outer_extend = 0

        # Clip outer extend to not create inverted tread area
        if width != 0 and outer_extend < 0 and abs(outer_extend) >= width:
            outer_extend = max(outer_extend, -width + 10)
        # Doesn't work if width hasn't been set manually, so another case is needed for default tire width (which is
        # calculated as a proportion of wheel diameter)
        elif width == 0 and outer_extend < 0 and top and bottom:
            outer_extend = max(outer_extend, int(-abs(bottom[1] - top[1]) /
                                                 self._default_tire_width_from_wheel_diameter_coefficient) + 10)

        # Make sure top is above bottom
        if top and bottom and top[1] > bottom[1]:
            # If top moved, apply floor to top
            if self.__prev_top and top[1] != self.__prev_top[1]:
                top = top[0], bottom[1]
            # If bottom moved, apply ceiling to bottom
            elif self.__prev_bottom and bottom[1] != self.__prev_bottom[1]:
                bottom = bottom[0], top[1]
            # Points have just been set and are wrong, just apply floor to top
            else:
                top = top[0], bottom[1]

            self.points[ord('t')] = top
            self.points[ord('b')] = bottom

        self.__prev_top = top
        self.__prev_bottom = bottom

        # All 3 main points have been set, ellipses can be constructed
        if top and bottom and right:
            # Ellipse has to be taller rather than wide
            cx, cy = (bottom[0] + top[0]) / 2, (bottom[1] + top[1]) / 2
            h = sqrt((bottom[0] - top[0])**2 + (bottom[1] - top[1])**2)
            w = 2 * sqrt((right[0] - cx)**2 + (right[1] - cy)**2)
            # Avoid division by zero
            w += (sys.float_info.epsilon if w == 0.0 else 0)
            # If ellipse is wider than taller, move right point closer to center (on circle perimeter) along its axis
            if int(w) > int(h):
                t = h/w
                right = int((1 - t) * cx + t * right[0]), int((1 - t) * cy + t * right[1])
                self.points[ord('r')] = right

            # Create ellipse from 3 main points
            ellipse = ellipse_from_points(top, bottom, right)

            # Ellipse is too small, can't proceed
            if ellipse.height == 0 or ellipse.width == 0:
                self.draw_only_annotation_points(image)
                return None

            # Create extractor
            angle = ellipse.angle
            extractor = Extractor(image, ellipse)
            # Overwrite angle (extractor sets ellipse upright)
            ellipse.angle = angle

            # Bounding ellipses
            outer, inner = extractor.get_tire_bounding_ellipses(tire_width=width, tire_sidewall=sidewall,
                                                                outer_extend=outer_extend)
            outer.angle = ellipse.angle
            inner.angle = ellipse.angle

            # Ellipses are too small, can't proceed
            if outer.height == 0 or outer.width == 0 or inner.height == 0 or inner.width == 0:
                self.draw_only_annotation_points(image)
                return None

            # Draw main ellipse
            cv2.ellipse(image, *ellipse.cv2_ellipse(), thickness=2, color=(0, 0, 255))
            cv2.circle(image, ellipse.get_center(), 5, (0, 0, 255), -1)
            # Draw lines between top and bottom and center and right point
            cv2.line(image, ellipse.get_center(), right, (255, 255, 0), thickness=1)
            cv2.line(image, top, bottom, (255, 255, 0), thickness=1)

            # Calculate start angle
            if start:
                y = (start[1] - outer.cy) / (outer.height / 2)
                y = np.clip(y, -1, 1)
                start_angle = degrees(asin(y))
            else:
                start_angle = self._default_start_angle

            # Calculate end angle
            if end:
                y = (end[1] - outer.cy) / (outer.height / 2)
                y = np.clip(y, -1, 1)
                end_angle = degrees(asin(y))
            else:
                end_angle = self._default_end_angle

            # Make sure start angle is above end angle
            if self.__prev_start != start:
                start_angle = min(start_angle, end_angle - 5)
            if self.__prev_end != end:
                end_angle = max(start_angle + 5, end_angle)
            start_angle = np.clip(start_angle, -90, 90)
            end_angle = np.clip(end_angle, -90, 90)
            if start_angle >= end_angle:
                start_angle = -90
                end_angle = 90

            # Drawing start and end angle points and lines between them marking the tread area
            start_a = outer.point_on_ellipse(start_angle)
            start_b = inner.point_on_ellipse(start_angle)
            end_a = outer.point_on_ellipse(end_angle)
            end_b = inner.point_on_ellipse(end_angle)
            start_a = int(start_a[0]), int(start_a[1])
            start_b = int(start_b[0]), int(start_b[1])
            end_a = int(end_a[0]), int(end_a[1])
            end_b = int(end_b[0]), int(end_b[1])

            # Draw lines marking tread area
            cv2.line(image, start_a, start_b, thickness=2, color=(0, 255, 0))
            cv2.line(image, end_a, end_b, thickness=2, color=(0, 255, 0))

            # Draw bounding ellipses
            cv2.ellipse(image, *outer.cv2_ellipse(), thickness=2, color=(0, 255, 255))
            cv2.ellipse(image, *inner.cv2_ellipse(-90, 90), thickness=2, color=(0, 255, 255))
            # Once again edges of tread area in different color
            cv2.ellipse(image, *outer.cv2_ellipse(start_angle, end_angle), thickness=2, color=(0, 255, 0))
            cv2.ellipse(image, *inner.cv2_ellipse(start_angle, end_angle), thickness=2, color=(0, 255, 0))

            # Move annotation points of start and end angles over to outer bounding ellipse
            # This is a real headache, since the same points are used to calculate the start angles earlier and
            # moving those incorrectly moves the angle, moving the points again and so on.
            outer_top = outer.point_on_ellipse(-90)
            outer_top = int(outer_top[0]), int(outer_top[1])
            outer_bottom = outer.point_on_ellipse(90)
            outer_bottom = int(outer_bottom[0]), int(outer_bottom[1])
            if not start:
                self.points[ord('u')] = start_a
            elif start[1] < outer_top[1]:
                self.points[ord('u')] = outer_top
            elif start[1] > end_a[1]:
                self.points[ord('u')] = start_a
            else:
                self.points[ord('u')] = start_a[0], self.points[ord('u')][1]
            if not end:
                self.points[ord('i')] = end_a
            elif end[1] > outer_bottom[1]:
                self.points[ord('i')] = outer_bottom
            elif end[1] < start_a[1]:
                self.points[ord('i')] = end_a
            else:
                self.points[ord('i')] = end_a[0], self.points[ord('i')][1]
            self.__prev_start = self.points[ord('u')]
            self.__prev_end = self.points[ord('i')]

            # All the constructed parameters
            result = extractor, ellipse, outer, inner, width, outer_extend, start_angle, end_angle
        else:
            # One of 3 main points missing, no ellipse could be constructed
            result = None

        self.draw_only_annotation_points(image)
        return result

    def draw_keypoints(self, image: np.ndarray) -> bool:
        """
        Draws 5 keypoints over image.

        Parameters
        ----------
        image : numpy.ndarray
            Image on which to draw keypoints on.

        Returns
        -------
        bool
            True when all points have been set and create an ellipse (with non-zero width or height).

            False otherwise.
        """

        top = self.points[ord('t')]
        bottom = self.points[ord('b')]
        right = self.points[ord('r')]
        sidewall = self.points[ord('s')]
        width = self.points[ord('w')]

        # Make sure top is above bottom
        if top and bottom and top[1] > bottom[1]:
            # If top moved, apply floor to top
            if self.__prev_top and top[1] != self.__prev_top[1]:
                top = top[0], bottom[1]
            # If bottom moved, apply ceiling to bottom
            elif self.__prev_bottom and bottom[1] != self.__prev_bottom[1]:
                bottom = bottom[0], top[1]
            # Points have just been set and are wrong, just apply floor to top
            else:
                top = top[0], bottom[1]

            self.points[ord('t')] = top
            self.points[ord('b')] = bottom

        self.__prev_top = top
        self.__prev_bottom = bottom

        # All 3 main points have been set, ellipses can be constructed
        if top and bottom and right:
            # Ellipse has to be taller rather than wide
            cx, cy = (bottom[0] + top[0]) / 2, (bottom[1] + top[1]) / 2
            h = sqrt((bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2)
            w = 2 * sqrt((right[0] - cx) ** 2 + (right[1] - cy) ** 2)
            # Avoid division by 0
            w += (sys.float_info.epsilon if w == 0.0 else 0)
            # If ellipse is wider than taller, move right point closer to center (on circle perimeter) along its axis
            if int(w) > int(h):
                t = h / w
                right = int((1 - t) * cx + t * right[0]), int((1 - t) * cy + t * right[1])
                self.points[ord('r')] = right

            # Create ellipse from 3 main points
            ellipse = ellipse_from_points(top, bottom, right)

            # Ellipse is too small, can't proceed
            if ellipse.height == 0 or ellipse.width == 0:
                self.draw_only_annotation_points(image)
                return False

            # Draw main ellipse
            cv2.ellipse(image, *ellipse.cv2_ellipse(), thickness=2, color=(0, 0, 255))
            cv2.circle(image, ellipse.get_center(), 3, (0, 0, 255), -1)
            # Draw lines between top and bottom and center and right point
            cv2.line(image, ellipse.get_center(), right, (255, 255, 0), thickness=1)
            cv2.line(image, top, bottom, (255, 255, 0), thickness=1)

            # Sidewall height and tire width are set
            if sidewall and width:
                # Find if sidewall point is closer to top or bottom point
                distance_to_top = abs(top[1] - sidewall[1])
                distance_to_bottom = abs(bottom[1] - sidewall[1])

                # Make sure sidewall is outside of main ellipse
                if top[1] - 10 < sidewall[1] < bottom[1] + 10:
                    if distance_to_top < distance_to_bottom:
                        sidewall = sidewall[0], top[1] - 10
                    else:
                        sidewall = sidewall[0], bottom[1] + 10
                    self.points[ord('s')] = sidewall
                # Make sure tire width is not negative
                if width[0] < right[0] + 10:
                    width = right[0] + 10, width[1]
                    self.points[ord('w')] = width

                # Create bounding box accounting for sidewall and tire width
                if distance_to_top < distance_to_bottom:
                    sidewall_height = abs(top[1] - sidewall[1])
                else:
                    sidewall_height = abs(bottom[1] - sidewall[1])
                tire_width = abs(right[0] - width[0])

                bbox = ellipse.bounding_box()
                top_left, bottom_right = bbox[0], bbox[1]

                tire_height = ellipse.height + sidewall_height * 2
                size_coefficient = tire_height / ellipse.height
                outer_width = int(ellipse.width * size_coefficient)

                top_left = top_left[0] - abs(outer_width - ellipse.width) // 2, top_left[1] - sidewall_height
                bottom_right = bottom_right[0] + tire_width, bottom_right[1] + sidewall_height

                # Draw bounding box
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), thickness=2)

                self.draw_only_annotation_points(image)
                return True

        self.draw_only_annotation_points(image)
        return False

    def annotate_keypoints(self) -> str:
        """
        Shows a cv2 window, uses same control scheme as Annotator.annotate(), except this method creates a JSON string
        consisting of bounding box (top left and bottom right coordinates) and keypoint coordinates.

        5 keypoints corresponding to the 'T', 'B', 'R', 'S' and 'W' keys must be labeled:

            'T' : Top of car wheel (without tire sidewall).

            'B' : Bottom of car wheel (without tire sidewall).

            'R' : Right (closest) side of car wheel (without tire sidewall). Image should be flipped the other way
                  around by 'F' if labeling the car's left side wheels. Refer to treadscan.extractor.CameraPosition.

            'S' : Top of tire (top of car wheel extended by tire sidewall height)

            'W' : Point on inner side of tire, gives the width

        Together, these keypoints can be used to create an ellipse, which defines the car wheel. This method can be used
        to create annotation for training a model for car wheel detection.

        Use spacebar to save 5 keypoints. You can create multiple ellipses by continuing with annotation after pressing
        spacebar. To verify your accuracy use 'P' to look at preview of what the extracted tread looks like. Preview
        can be hidden using backspace. Use enter to submit keypoints (ellipses).

        Returns
        -------
        str
            Empty if annotation is canceled (escape).

            JSON string with bounding box and keypoints if submitted (enter).
        """

        self.points = {
            ord('t'): None,  # Top
            ord('b'): None,  # Bottom
            ord('r'): None,  # Right
            ord('s'): None,  # Tire sidewall
            ord('w'): None   # Tire width
        }

        def mouse_callback(event, x, y, flags, param):
            # Save current mouse position over image
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = x, y

        submitted = False
        bounding_boxes = []
        keypoints = []

        def save_keypoints():
            """Save annotated keypoints"""

            top = self.points[ord('t')]
            bottom = self.points[ord('b')]
            right = self.points[ord('r')]
            sidewall = self.points[ord('s')]
            width = self.points[ord('w')]

            # Reset keypoints
            self.points = {
                ord('t'): None,  # Top
                ord('b'): None,  # Bottom
                ord('r'): None,  # Right
                ord('s'): None,  # Tire sidewall
                ord('w'): None   # Tire width
            }
            # Keep drawn ellipse
            self.image = image

            # Save points and bounding box
            top = int(top[0] / self.scale), int(top[1] / self.scale), 1
            bottom = int(bottom[0] / self.scale), int(bottom[1] / self.scale), 1
            right = int(right[0] / self.scale), int(right[1] / self.scale), 1
            sidewall = int(sidewall[0] / self.scale), int(sidewall[1] / self.scale), 1
            width = int(width[0] / self.scale), int(width[1] / self.scale), 1

            ellipse = ellipse_from_points(top, bottom, right)
            bbox = ellipse.bounding_box()
            # Extend bounding box to entire tire
            top_left, bottom_right = bbox[0], bbox[1]
            distance_to_top = abs(top[1] - sidewall[1])
            distance_to_bottom = abs(bottom[1] - sidewall[1])
            if distance_to_top < distance_to_bottom:
                sidewall_height = abs(top[1] - sidewall[1])
            else:
                sidewall_height = abs(bottom[1] - sidewall[1])
            tire_width = abs(right[0] - width[0])
            tire_height = ellipse.height + sidewall_height * 2
            size_coefficient = tire_height / ellipse.height
            outer_width = int(ellipse.width * size_coefficient)
            top_left = top_left[0] - abs(outer_width - ellipse.width) // 2, top_left[1] - sidewall_height
            bottom_right = bottom_right[0] + tire_width, bottom_right[1] + sidewall_height

            keypoints.append([top, bottom, right, sidewall, width])
            bounding_boxes.append([*top_left, *bottom_right])

        tread = None
        while True:
            # Start with clean image
            image = self.image.copy()
            # If tread preview exists, show it in upper right corner
            if tread is not None:
                h1, w1 = image.shape[0], image.shape[1]
                h2, w2 = tread.shape[0], tread.shape[1]
                if h2 > h1 or w2 > w1:
                    # If preview wouldn't fit, scale it down
                    scale = min(w1 / w2, h1 / h2)
                    scale = min(scale, 1)
                    tread = scale_image(tread, scale)
                    h2, w2 = tread.shape[0], tread.shape[1]

                image[0:h2, w1 - w2:w1, :] = tread[0:h2, 0:w2, :]
            # Draw points
            success = self.draw_keypoints(image)

            # Wait for user input
            cv2.imshow('Image', image)
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Image', mouse_callback)
            key = cv2.waitKey(30)

            # If user closed window or pressed escape
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                break

            # Enter to submit
            elif key == 13:
                if success:
                    save_keypoints()
                submitted = True
                break

            # P to show tread preview
            elif key == ord('p'):
                if success:
                    top = self.points[ord('t')]
                    bottom = self.points[ord('b')]
                    right = self.points[ord('r')]
                    sidewall = self.points[ord('s')]
                    width = self.points[ord('w')]

                    ellipse = ellipse_from_points(top, bottom, right)
                    sidewall = int(np.sqrt((top[0] - sidewall[0]) ** 2 + (top[1] - sidewall[1]) ** 2))

                    extractor = Extractor(self.image, ellipse)
                    bounding_ellipses = extractor.get_tire_bounding_ellipses(tire_sidewall=sidewall, tire_limit=width)
                    tread = extractor.extract_tread(tire_bounding_ellipses=bounding_ellipses)

            # Backspace to hide preview
            elif key == 8 and tread is not None:
                tread = None

            # An annotating key was pressed
            elif key != -1 and key in self.points.keys():
                self.points[key] = self.mouse_pos

            # F to flip image
            elif key == ord('f'):
                self.image = cv2.flip(self.image, 1)
                self.flipped = not self.flipped

            # Spacebar to save keypoints
            elif key == 32 and success:
                save_keypoints()

        cv2.destroyAllWindows()

        if submitted and keypoints and bounding_boxes:
            annotations = {'bboxes': bounding_boxes, 'keypoints': keypoints}
            return json.dumps(annotations)
        else:
            return ''

    def annotate(self) -> Union[None, tuple]:
        """
        Shows a cv2 window that responds to key inputs, allowing to create bounding ellipses and look at preview of
        extracted tread.

        Controls:
            Escape or closing window: cancel annotation and close window, returns None.

            Enter: submit annotation and close window, returns annotated parameters.

            Spacebar: show preview of extracted tread if possible.

            Backspace: hide preview of extracted tread.

            F: flip image horizontally (mirror).

        Annotation points:
            T: Top of main ellipse.

            B: Bottom of main ellipse.

            R: Right side of main ellipse (or left side, width is determined as Euclidean distance between this point
            and main ellipse center).

            S: Sidewall height (adjusts height of outer ellipse). On the bottom or top of tire.

            O: Outer ellipse extension (move outer ellipse left or right).

            W: Width of tire (move inner ellipse left or right)

            U: Starting angle of tread extraction (top of tread).

            I: Ending angle of tread extraction (bottom of tread).

        Returns
        -------
        None
            If annotation was canceled.

        (treadscan.Ellipse, treadscan.Ellipse, treadscan.Ellipse, int, float, float, bool)
            Tuple of main ellipse, outer and inner bounding ellipses, tread width, start and end extraction angle and
            boolean whether image is flipped (mirrored).
        """

        def mouse_callback(event, x, y, flags, param):
            # Save current mouse position over image
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = x, y

        # Preview of tread extraction
        tread = None
        while True:
            # Start with clean image
            image = self.image.copy()
            # Draw annotated points and structures over image, get created structures
            parameters = self.draw_points(image)

            # If tread preview exists, show it in upper right corner
            if tread is not None:
                h1, w1 = image.shape[0], image.shape[1]
                h2, w2 = tread.shape[0], tread.shape[1]
                if h2 > h1 or w2 > w1:
                    # If preview wouldn't fit, scale it down
                    scale = min(w1 / w2, h1 / h2)
                    scale = min(scale, 1)
                    tread = scale_image(tread, scale)
                    h2, w2 = tread.shape[0], tread.shape[1]

                image[0:h2, w1 - w2:w1, :] = tread[0:h2, 0:w2, :]

            # Finally, draw annotated image, wait for user input
            cv2.imshow('Image', image)
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Image', mouse_callback)
            key = cv2.waitKey(30)

            # If user closed window or pressed escape
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                parameters = None
                break

            # Enter to submit
            elif key == 13 and parameters is not None:
                break

            # An annotating key was pressed
            elif key != -1 and key in self.points.keys():
                self.points[key] = self.mouse_pos

            # F to flip image
            elif key == ord('f'):
                self.image = cv2.flip(self.image, 1)
                self.flipped = not self.flipped

            # Spacebar to show preview
            elif key == 32 and parameters is not None:
                ex, _, outer, inner, width, extend, start_angle, end_angle = parameters
                tread = ex.extract_tread(tire_width=abs(width+extend), tire_bounding_ellipses=(outer, inner),
                                         start=start_angle, end=end_angle, cores=1)
                # Increase contrast and brightness
                tread = cv2.addWeighted(tread, 2, tread, 0, 50)

            # Backspace to hide preview
            elif key == 8 and tread is not None:
                tread = None

        cv2.destroyAllWindows()

        if parameters:
            # If user submitted annotation
            _, main_ellipse, outer_ellipse, inner_ellipse, width, extend, start_angle, end_angle = parameters

            # Scale parameters back to original size
            main_ellipse.cx = int(main_ellipse.cx / self.scale)
            main_ellipse.cy = int(main_ellipse.cy / self.scale)
            main_ellipse.width = int(main_ellipse.width / self.scale)
            main_ellipse.height = int(main_ellipse.height / self.scale)
            outer_ellipse.cx = int(outer_ellipse.cx / self.scale)
            outer_ellipse.cy = int(outer_ellipse.cy / self.scale)
            outer_ellipse.width = int(outer_ellipse.width / self.scale)
            outer_ellipse.height = int(outer_ellipse.height / self.scale)
            inner_ellipse.cx = int(inner_ellipse.cx / self.scale)
            inner_ellipse.cy = int(inner_ellipse.cy / self.scale)
            inner_ellipse.width = int(inner_ellipse.width / self.scale)
            inner_ellipse.height = int(inner_ellipse.height / self.scale)
            width = int(width / self.scale)
            extend = int(extend / self.scale)

            tread_width = abs(width + extend)

            return main_ellipse, outer_ellipse, inner_ellipse, tread_width, start_angle, end_angle, self.flipped
        else:
            # Annotation was canceled
            return None
