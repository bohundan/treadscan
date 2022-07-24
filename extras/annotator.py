"""
This module contains the Annotator class which uses OpenCV's

Annotator class also contains a method for annotating keypoints of the tire.
"""

from datetime import datetime
import json
from typing import Optional
import sys

import cv2
import numpy as np
from treadscan.extractor import TireModel
from treadscan.utilities import *


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

    mouse_pos : (int, int)
        Position of mouse over image (X and Y coordinates).

    points : dict[int, Any]
        Contains position of all the annotation points.

    Methods
    -------
    draw_only_annotation_points(image: np.ndarray)
        Draws just the keypoints on image.

    draw(image: np.ndarray)
        Draws other structures on image (Ellipse, TireModel, bounding box).

    annotate_keypoints()
        Shows a GUI, allowing user to put keypoints in image and construct TireModels.
    """

    def __init__(self, image: np.ndarray, max_width: int, max_height: int):
        """
        Parameters
        ----------
        image : np.ndarray
            Original grayscale (or BGR) image which will be annotated.

        max_width : int
            Maximum width of annotation window (without preview).

        max_height : int
            Maximum height of annotation window (without preview).
        """

        self.scale = min(max_width / image.shape[1], max_height / image.shape[0])
        self.scale = min(self.scale, 1)

        self.image = scale_image(image, self.scale)
        # Convert grayscale to BGR
        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            self.image = image.copy()
        self._original_image = self.image.copy()

        self.tire_model = TireModel((self.image.shape[0], self.image.shape[1]))

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

        # Hidden attributes for less wonky behaviour of some annotation points to stop movement of multiple points as a
        # reaction to one point changing
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

        point_size = 5

        # Unused points are drawn in top left corner in one column
        y_pos = 8
        for key, value in self.points.items():
            if value is not None:
                # Point is set, draw it where it is set
                cv2.circle(image, value, point_size, (0, 128, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                point = (value[0] - 4, value[1] + 4)
                cv2.putText(image, chr(key), point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0),
                            thickness=1, lineType=cv2.LINE_AA)
            else:
                # Point is not set, draw it at the bottom of the column
                cv2.circle(image, (4, y_pos), point_size, (0, 128, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                cv2.putText(image, chr(key), (0, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                y_pos += 16

    def draw(self, image: np.ndarray) -> bool:
        """
        Draws keypoints and models over image.

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

        # All 3 main points have been set, ellipse can be constructed
        if top and bottom and right:
            # Ellipse has to be taller rather than wider (use Euclidean distance)
            cx, cy = (bottom[0] + top[0]) / 2, (bottom[1] + top[1]) / 2
            h = euclidean_dist(top, bottom)
            w = 2 * euclidean_dist(right, (cx, cy))
            # Avoid division by 0
            w += (sys.float_info.epsilon if w == 0.0 else 0)
            # If ellipse is wider than taller, move right point closer to center (on circle perimeter) along its axis
            if int(w) > int(h):
                t = h / w
                right = int((1 - t) * cx + t * right[0]), int((1 - t) * cy + t * right[1])
                self.points[ord('r')] = right
            # Make sure that ellipse isn't a line (third point doesn't lie between top and bottom)
            epsilon = 1
            if -epsilon < (euclidean_dist(top, right) +
                           euclidean_dist(right, bottom) -
                           euclidean_dist(top, bottom)) < epsilon:
                self.points[ord('r')] = right

            # Create ellipse from 3 main points
            ellipse = ellipse_from_points(top, bottom, right)

            # Ellipse is too small, can't proceed
            if ellipse.height == 0 or ellipse.width == 0:
                self.draw_only_annotation_points(image)
                return False

            # Draw main ellipse
            cv2.ellipse(image, *ellipse.cv2_ellipse(), thickness=2, color=(0, 0, 255), lineType=cv2.LINE_AA)

            # Sidewall height and tire width are also set
            if sidewall and width:
                # Create tire model
                self.tire_model.from_keypoints(top, bottom, right, sidewall, width)
                # Draw bounding box
                top_left, bottom_right = self.tire_model.bounding_box()
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                # Draw tire model
                self.tire_model.draw(image, (0, 255, 255), 2, cv2.LINE_AA)
                # Draw keypoints
                self.draw_only_annotation_points(image)

                return True

        # Some keypoints are not set yet, draw only keypoints
        self.draw_only_annotation_points(image)
        return False

    def annotate_keypoints(self) -> str:
        """
        Shows a cv2 window, uses same control scheme as Annotator.annotate(), except this method creates a JSON string
        consisting of bounding box (top left and bottom right coordinates) and keypoint coordinates.

        5 keypoints corresponding to the 'T', 'B', 'R', 'S' and 'W' keys must be labeled:

            'T' : Top of vehicle's rim (without tire sidewall).

            'B' : Bottom of vehicle's rim (without tire sidewall).

            'R' : Right (closest) side of vehicle's rim (without tire sidewall).

            'S' : Top of tire (top of vehicle's rim extended by tire sidewall height)

            'W' : Point on inner side of tire, gives the width

        Together, these keypoints can be used to create an ellipse, which defines the vehicle's tire. This method can be
        used to create annotation for training a model for vehicle tire detection.

        Use enter to save keypoints. You can create multiple ellipses by continuing with annotation after pressing
        the 'N' key. To verify your accuracy use 'spacebar to look at preview of what the extracted tread looks like.
        Preview can be hidden using backspace.

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

            # Draw permanent transparent rectangle over original image to mark as saved
            top_left, bottom_right = self.tire_model.bounding_box()
            rect = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            colored = np.zeros(shape=rect.shape, dtype=np.uint8)
            cv2.rectangle(colored, (0, 0), (colored.shape[1], colored.shape[0]), color=(255, 255, 0),
                          thickness=cv2.FILLED, lineType=cv2.LINE_AA)
            rect = cv2.addWeighted(rect, 0.5, colored, 0.5, 1.0)
            self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = rect

            # Save keypoints together with third parameter (visibility) and bounding box
            top = int(top[0] / self.scale), int(top[1] / self.scale), 1
            bottom = int(bottom[0] / self.scale), int(bottom[1] / self.scale), 1
            right = int(right[0] / self.scale), int(right[1] / self.scale), 1
            sidewall = int(sidewall[0] / self.scale), int(sidewall[1] / self.scale), 1
            width = int(width[0] / self.scale), int(width[1] / self.scale), 1
            # Rescale bounding box also
            top_left = int(np.floor(top_left[0] / self.scale)), int(np.floor(top_left[1] / self.scale))
            bottom_right = int(np.floor(bottom_right[0] / self.scale)), int(np.floor(bottom_right[1] / self.scale))

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
            success = self.draw(image)

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

            # Space to show tread preview
            elif key == 32:
                if success:
                    try:
                        tread = self.tire_model.unwrap(self._original_image)

                        # Improve preview by equalizing histogram
                        if len(tread.shape) == 3:
                            tread = cv2.cvtColor(tread, cv2.COLOR_BGR2GRAY)
                        tread = equalize_grayscale(tread)
                        tread = cv2.cvtColor(tread, cv2.COLOR_GRAY2BGR)
                    except RuntimeError as e:
                        print(f'Created tire model is not possible. "{e}"')

            # Backspace to hide preview
            elif key == 8 and tread is not None:
                tread = None

            # An annotating key was pressed
            elif key != -1 and key in self.points.keys():
                self.points[key] = self.mouse_pos

            # N to save keypoints
            elif key == ord('n') and success:
                save_keypoints()
                tread = None

        cv2.destroyAllWindows()

        if submitted and keypoints and bounding_boxes:
            annotations = {'bboxes': bounding_boxes, 'keypoints': keypoints}
            return json.dumps(annotations)
        else:
            return ''
