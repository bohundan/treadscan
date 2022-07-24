"""
This module is used for detecting stopped vehicle(s) from camera footage.

For example if used on footage of vehicles stopping in front of traffic lights, Detector.detect() should only yield one
image per one stopped vehicle. This is achieved by comparing every frame against static background image, when there's
a low match between the two, the frame likely contains another object (a vehicle). There would be more sequential frames
containing an object, so the next step is to compare each two consequent frames with each other. If they're nearly
identical it can be assumed that there's no movement (the longer the time passed between two frames the more obvious
the difference will be if there is movement).

But there can be many such consequent frames, as vehicles might be stopped waiting at the traffic lights for some time,
so to obtain just one image, these frames (containing stopped vehicles) are compared all against each other by measuring
their Laplacian variance, which in short ensures that the best focused frame will be the one detected.
These methods combined create a state-machine of sorts

START : No vehicle

        -> START (current frame matches background)
        -> 1     (current frame doesn't match background)
        -> END   (last frame, end of footage)

1 : Object in frame

        -> START (current frame matches background)
        -> 1     (current frame doesn't match background and doesn't match previous frame)
        -> 2     (current frame doesn't match background and matches previous frame)
        -> END   (last frame, end of footage)

2 : Object in frame not moving
        Remember current frame and its Laplacian variance

        -> 1     (current frame doesn't match background and doesn't match previous frame)
                 Yield best frame of stopped vehicle, vehicle now started moving
        -> 2     (current frame doesn't match background and matches previous frame)
                 If Laplacian variance of current frame is better than previous best, new best frame detected
        -> END   (last frame, end of footage)
                 Yield best frame (footage ended before object started moving)

END : End of footage

Input can be a collection of images (inside folder, correctly sorted) or a video/stream, refer to cv2.VideoCapture() to
find out more about supported video/stream formats. Supported image formats are those supported by cv2.imread().
"""

from collections import OrderedDict
from enum import Enum
from os import listdir
from os.path import isfile, isdir, join
from typing import Generator, Optional

import cv2
import numpy as np

from .utilities import scale_image, subsample_hash


class InputType(Enum):
    """
    Enum class specifying types of input for Detector class.

    IMAGE_FOLDER :
        Path to folder which contains only images (sorted alphabetically). Alternative to having to encode them into a
        video, which would just be decoded again, as footage is processed frame by frame.

    VIDEO :
        Path to video file.

    STREAM :
        Path or URL, see OpenCV's VideoCapture class. Supports video files, image sequences or cameras.
    """

    IMAGE_FOLDER = 1
    VIDEO = 2
    STREAM = 2


class FrameExtractor:
    """
    Class for extracting GRAYSCALE frames from video or folder of images.

    Attributes
    ----------
    input_type : treadscan.InputType
        Folder of images (has to be sorted alphabetically) or video/stream (cv2.VideoCapture compatible).

    frame_index : int
        Number of frame (frame returned by next_frame), starts at 1 (first frame is frame 1).

    files : list of str
        Sorted list of paths to individual images (when using image folder input).

    video : cv2.VideoCapture
        Loaded video/stream (when using video input).

    end : Optional[int]
        Index of last frame to be extracted (when using treadscan.InputType.IMAGE_FOLDER).

    Methods
    -------
    next_frame()
        Returns next frame in sequence, None if at the end.
    """

    def __init__(self, input_path: str, input_type: InputType):
        """
        Parameters
        ----------
        input_path : str
           Path to video/stream or folder.

        input_type : treadscan.InputType

        Raises
        ------
        ValueError
            If video/stream or path to folder doesn't exist or isn't readable.
        """

        # Load video
        if input_type == InputType.VIDEO:
            self.video = cv2.VideoCapture(input_path)
            if not self.video.isOpened():
                raise ValueError('Failed opening video file/stream.')

        # Load folder
        elif input_type == InputType.IMAGE_FOLDER:
            if not isdir(input_path):
                raise ValueError('Image folder does not exist or is not readable.')
            self.files = [f'{input_path}/{file}' for file in listdir(input_path) if isfile(join(input_path, file))]
            self.files.sort()

        self.input_type = input_type
        self.frame_index = 0

        self.end = None

    def next_frame(self) -> Optional[np.ndarray]:
        """
        Gets the next frame from input, converts it to grayscale and returns it.

        Returns
        -------
        numpy.ndarray
            The next frame as grayscale image.
        None
            When at the end of sequence (end of video/stream).
        """

        # Folder of images
        if self.input_type == InputType.IMAGE_FOLDER:
            if self.frame_index >= len(self.files):
                return None
            elif self.end and self.frame_index > self.end:
                return None
            else:
                # Increment and return next frame
                self.frame_index += 1
                return cv2.imread(self.files[self.frame_index - 1], cv2.IMREAD_GRAYSCALE)

        # Video file
        elif self.input_type == InputType.VIDEO:
            ret, frame = self.video.read()

            if not ret:
                # At the end
                self.video.release()
                return None
            else:
                # Increment and return next frame
                self.frame_index += 1
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def set_folder_bounds(self, start: Optional[int] = None, end: Optional[int] = None):
        """
        Set start and/or end index of images in image folder.

        Parameters
        ----------
        start : Optional[int]
            Index of first frame.

        end : Optional[int]
            Index of last frame.

        Raises
        ------
        RuntimeError
            When trying to set bounds but FrameExtractor doesn't have folder as treadscan.InputType.

        ValueError
            When end > start.
        """

        if self.input_type != InputType.IMAGE_FOLDER:
            raise RuntimeError('Cannot set start or end index for this input type.')

        if start is not None and end is not None and start > end:
            raise ValueError('Start index is greater than end index.')

        self.end = end
        if start is not None:
            self.frame_index = start


class BackgroundSubtractorSimple(cv2.BackgroundSubtractor):
    """
    Simple background subtractor immune to *sleeping person* phenomenon (an object that stops moving won't become a part
    of the background). This subtractor doesn't adapt to changing background and needs to be provided with one manually.
    Only works with grayscale images.

    Same interface as OpenCV's BackgroundSubtractor
    (https://docs.opencv.org/3.4/d7/df6/classcv_1_1BackgroundSubtractor.html).

    Attributes
    ----------
    background_sample : numpy.ndarray
        Image of empty background (without foreground objects present).

    intensity_threshold : int
        Minimal difference between pixel values to be subtracted. If the difference is smaller than this threshold, the
        pixel is regarded as a *background pixel*.

    Methods
    -------
    apply(image: numpy.ndarray)
        Returns mask of foreground objects.

    getBackgroundImage()
        Returns background image.
    """

    def __init__(self, background_sample: np.ndarray, intensity_threshold: int = 50):
        """
        Parameters
        ----------
        background_sample : numpy.ndarray
            Image of empty background (without foreground objects present).

        intensity_threshold : int
            Minimal difference between pixel values to be subtracted. If the difference is smaller than this threshold,
            the pixel is regarded as a *background pixel*.

        Raises
        ------
        ValueError
            When background_sample isn't grayscale.
        """

        super().__init__()
        if len(background_sample.shape) != 2:
            raise ValueError('BackgroundSubtractorSimple only works with grayscale images.')

        self.background_sample = background_sample
        self.intensity_threshold = intensity_threshold

        self._cached_background = self.background_sample

    def apply(self, image: np.ndarray, fgmask=None, learningRate: float = -1) -> np.ndarray:
        """
        Remove background from image.

        Parameters
        ----------
        image : numpy.ndarray
            Image from which to remove background.

        fgmask : None

        learningRate : float
            Does nothing in the case of BackgroundSubtractorSimple.

        Returns
        -------
        numpy.ndarray
            Binary mask of foreground objects.

        Raises
        ------
        RuntimeError
            If image isn't grayscale.
        """

        if len(image.shape) != 2:
            raise RuntimeError('BackgroundSubtractorSimple only works with grayscale images.')

        if image.shape != self._cached_background.shape:
            scale = image.shape[0] / self._cached_background.shape[0]
            self._cached_background = scale_image(self.background_sample, scale)

        difference = abs(self._cached_background.astype(np.intc) - image.astype(np.intc)).astype(np.uint8)
        mask = np.greater(difference, self.intensity_threshold).astype(np.uint8) * 255

        return mask

    def getBackgroundImage(self, backgroundImage=None) -> np.ndarray:
        """
        Implemented for compatibility.

        Parameters
        ----------
        backgroundImage : None

        Returns
        -------
        numpy.ndarray
            Background image.
        """

        return self.background_sample


class Detector:
    """
    Detects presence and motion of a vehicle from footage, yielding 1 image per 1 stopped vehicle.

    Input can either be a folder full of images (has to be sorted) or a video/stream.

    Attributes
    ----------
    backsub : cv2.BackgroundSubtractor
        Instance of a background subtractor.

    frame_extractor : treadscan.FrameExtractor
        Frame extractor used to get frames from input footage for detection.

    vehicle_threshold : float
        Minimum percentage of 'not background' pixels to evaluate image as containing a vehicle.

    motion_intensity_threshold : int
        Difference between pixels needed to be classified as 'motion' (between 2 subsequent images).

    motion_threshold : float
        Minimum percentage of 'moving' pixels to evaluate 2 subsequent images as having motion.

    focus_intensity_threshold : int
        Compare focus (or blurriness, by edge detection) in parts of image that are darker than this threshold.

    lookup_table : collections.OrderedDict
        HashTable with hashes of images and their Laplacian variance. Used when comparing blurriness of frames to avoid
        needlessly calculating the same variance of the same image multiple times.

    lookup_table_max_size : int
        Maximum size lookup_table can grow to before old values start being purged to limit memory usage.

    Methods
    -------
    detect(scale: float, window: int)
        Generator which yields one image per one stopped vehicle. Uses a window, in which it detects vehicle presence
        and whether it is moving or not. Window opens when a vehicle is detected as stopped and closes when vehicle
        starts moving again. The generator then yields the best focused (least blurry) frame from this window.

    set_params(vehicle_threshold: Optional[float], motion_threshold: Optional[float],
               motion_intensity_threshold: Optional[int], focus_intensity_threshold: Optional[int]):
       Sets detection parameters.

    vehicle_mask(image: numpy.ndarray)
        Returns binary mask of vehicle.

    motion_mask(image1: numpy.ndarray, image2: numpy.ndarray)
        Returns binary mask of motion between two images.

    cached_laplacian_var(image: numpy.ndarray, strategy: str, scale: float)
        Calculates Laplacian variance of image.

    compare_image_focus(image1: numpy.ndarray, image2: numpy.ndarray)
        Compares *blurriness* of the two images, returns the less blurry one.

    get_full_detection_data(scale: float, window: int)
        Generate datasets for analysis of detection parameters.
    """

    def __init__(self, backsub: cv2.BackgroundSubtractor, frame_extractor: FrameExtractor):
        """
        Parameters
        ----------
        backsub : cv2.BackgroundSubtractor
            Instance of background subtractor.

        frame_extractor : FrameExtractor
            Input video/stream from which to detect stopped vehicles.
        """

        self.backsub = backsub
        self.frame_extractor = frame_extractor

        # Minimum percentage of 'not background' pixels to evaluate image as containing a vehicle
        self.vehicle_threshold = 0.2

        # Difference between pixels needed to be classified as 'motion' (between 2 subsequent images)
        self.motion_intensity_threshold = 10
        # Minimum percentage of 'moving' pixels to evaluate 2 subsequent images as having motion
        self.motion_threshold = 0.02

        # Compare focus (or blurriness, by edge detection) in parts of image that are darker than this threshold
        self.focus_intensity_threshold = 50

        # Structure used for caching some results that can be reused to speed up processing time
        self.lookup_table = OrderedDict()
        self.lookup_table_max_size = 1024

    @staticmethod
    def _calc_kernel_size(image_width: int, image_height: int) -> (int, int):
        """
        Calculate odd kernel size to use for blurring images (for processing footage).

        Parameters
        ----------
        image_width : int

        image_height : int

        Returns
        -------
        (int, int)
            Tuple of integers (square kernel of odd size).
        """

        # 1/50 of the smaller dimension
        kernel_size = min(image_width, image_height) // 50
        # Ensure oddness
        kernel_size += (1 if kernel_size % 2 == 0 else 0)

        return kernel_size, kernel_size

    @staticmethod
    def _prep_image(image: np.ndarray, scale: float, blur_kernel: tuple):
        """
        Scale and blur image for processing.

        Parameters
        ----------
        image : numpy.ndarray

        scale : float

        blur_kernel : (int, int)

        Returns
        -------
        numpy.ndarray
            Scaled and blurred image.
        """

        image = scale_image(image, scale)
        image = cv2.blur(image, blur_kernel)

        return image

    def detect(self, scale: float = 0.25, window: int = 50) -> Generator[np.ndarray, None, None]:
        """
        Generator which yields one frame for footage per one stopped vehicle.

        Parameters
        ----------
        scale : float
            Rescales footage down (to speed up processing).

        window : int
            Number of frames that are also considered (starts with first frame where stopped vehicle is detected).
            Compensates for small movements and noise in footage that could otherwise cause one vehicle to yield
            multiple images.
            The higher the framerate, the longer window is recommended.

        Yields
        ------
        numpy.ndarray
            One image per one stopped vehicle from footage.

        Raises
        ------
        ValueError
            If scale factor is not between 1 and 0.

        RuntimeError
            If failed to fetch first frame from footage.
        """

        if not 0 < scale <= 1:
            raise ValueError('Invalid scale factor (must be between 1 and 0).')

        last_hit = window
        best_frame = None

        # Acquire first frame
        prev_frame = self.frame_extractor.next_frame()
        if prev_frame is None:
            raise RuntimeError('Failed fetching first frame from input. Possibly reached the end.')

        # Image processing context
        height, width = prev_frame.shape
        height, width = int(height * scale), int(width * scale)
        blur_kernel = self._calc_kernel_size(width, height)

        # Scale image down
        prev_frame_pp = self._prep_image(prev_frame, scale, blur_kernel)

        curr_frame = self.frame_extractor.next_frame()
        while curr_frame is not None:
            # Process next frame
            curr_frame_pp = self._prep_image(curr_frame, scale, blur_kernel)

            vehicle_presence = np.count_nonzero(self.vehicle_mask(curr_frame_pp)) / (width * height)
            vehicle_motion = np.count_nonzero(self.motion_mask(curr_frame_pp, prev_frame_pp)) / (width * height)

            # Open or lengthen window when stopped vehicle is detected
            if vehicle_presence > self.vehicle_threshold and vehicle_motion < self.motion_threshold:
                last_hit = 0
            elif last_hit < window:
                last_hit += 1

            if last_hit < window:
                # Get the best focused frame in window (only frames where car is present and not moving are considered)
                if last_hit == 0:
                    if best_frame is None:
                        best_frame = curr_frame
                    else:
                        best_frame = self.compare_image_focus(curr_frame, best_frame)
            elif best_frame is not None:
                # Window has closed, yield best focused frame (and unset it to wait for next opened window)
                yield best_frame
                best_frame = None

            # Next frame
            prev_frame_pp = curr_frame_pp
            curr_frame = self.frame_extractor.next_frame()

        # Make sure to yield last frame if footage ends with window open
        if best_frame is not None:
            yield best_frame

    def set_params(self, vehicle_threshold: Optional[float] = None, motion_threshold: Optional[float] = None,
                   motion_intensity_threshold: Optional[int] = None, focus_intensity_threshold: Optional[int] = None):
        """
        Set parameters used to detect stopped vehicles(s) from footage. If parameter is None then it remains unchanged.

        Parameters
        ----------
        vehicle_threshold : float
            Minimum percentage of 'not background' pixels to evaluate image as containing a vehicle. (Minimum size of
            vehicle in image).

        motion_threshold : float
            Minimum percentage of 'moving' pixels to evaluate 2 subsequent images as having motion.

        motion_intensity_threshold : int
            Difference between pixels needed to be classified as 'motion' (between 2 subsequent images).

        focus_intensity_threshold : int
            Compare focus (or blurriness) of images using edge detection (variance of the Laplacian operator) in parts
            that are darker than this threshold.

        Raises
        ------
        ValueError
            When provided parameter is out of range or if path to image doesn't exist or is not readable.
        """

        if vehicle_threshold is not None:
            if not 0 <= vehicle_threshold <= 1:
                raise ValueError('Vehicle threshold out of range (must be between 0 and 1)')
            self.vehicle_threshold = vehicle_threshold

        if motion_threshold is not None:
            if not 0 <= motion_threshold <= 1:
                raise ValueError('Motion threshold out of range (must be between 0 and 1)')
            self.motion_threshold = motion_threshold

        if motion_intensity_threshold is not None:
            if not 0 <= motion_intensity_threshold <= 255:
                raise ValueError('Motion intensity threshold out of range (must be between 0 and 255)')
            self.motion_intensity_threshold = motion_intensity_threshold

        if focus_intensity_threshold is not None:
            if not 0 <= focus_intensity_threshold <= 255:
                raise ValueError('Focus intensity threshold out of range (must be between 0 and 255)')
            self.focus_intensity_threshold = focus_intensity_threshold

    def vehicle_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detects vehicle in image by subtracting background. Returns binary mask of vehicle.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        numpy.ndarray
            Binary mask of vehicle (foreground object).
        """

        return self.backsub.apply(image)

    def motion_mask(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Determines whether there is motion between two images. Returns binary mask of detected motion.

        Uses self.motion_intensity_threshold as the threshold of minimum difference of pixel values to evaluate them as
        "having changed" - moving, and self.motion_threshold as the threshold of percentage of pixels that are "moving"
        to evaluate the images as having motion.

        Parameters
        ----------
        image1 : numpy.ndarray

        image2 : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Binary mask of detected motion.
        """

        difference = abs(image1.astype(np.intc) - image2.astype(np.intc)).astype(np.uint8)
        mask = np.greater(difference, self.motion_intensity_threshold).astype(np.uint8) * 255

        return mask

    def cached_laplacian_var(self, image: np.ndarray, strategy: str = 'aggressive', scale: float = 0.25):
        """
        Calculates variance of the Laplacian operator on DARK parts of the image (darker than
        self.focus_intensity_threshold). Uses hashtable to cache results.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        strategy : str
            Strategy to employ when clearing hashtable (making space for new results after it has been filled).

            'aggressive' : Clean entire hashtable (default).

            'lazy' : Remove exactly 1 entry (the oldest one).

            Any other value defaults to 'aggressive' strategy.

        scale : float
            Resize image before edge detection (faster for large images).

        Returns
        -------
        float
            Variance of Laplacian operator.

        Notes
        -----
        Uses a hashtable to store previous results, as the main method, Detector.detect() may use Laplacian variance
        to compare *blurriness* of one image multiple times. Using a cached results avoids having to compute the same
        value for the same image multiple times, improving performance.
        """

        image_hash = subsample_hash(image)

        # If result is not cached, it has to be computed
        if image_hash not in self.lookup_table.keys():
            if len(self.lookup_table) >= self.lookup_table_max_size:
                if strategy == 'lazy':
                    self.lookup_table.popitem(last=False)
                else:
                    self.lookup_table = OrderedDict()

            # Compute Laplacian variance (of dark parts of resized image)
            small = scale_image(image, scale)
            mask = np.less(small, self.focus_intensity_threshold).astype(np.uint8)
            dark = small * mask
            self.lookup_table[image_hash] = cv2.Laplacian(dark, cv2.CV_64F).var()

        # Return cached Laplacian variance of image
        return self.lookup_table[image_hash]

    def compare_image_focus(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compare *blurriness* between two images, return the less blurry one.

        Parameters
        ----------
        image1 : numpy.ndarray

        image2 : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            The less blurry image of the two.
        """

        lap_var1 = self.cached_laplacian_var(image1)
        lap_var2 = self.cached_laplacian_var(image2)

        if lap_var1 < lap_var2:
            return image2
        else:
            return image1

    def get_full_detection_data(self, scale: float = 0.25, window: int = 50) -> (list, list, list, list):
        """
        Create datasets from footage for analysis, visualisation or easier parameter selection (picking the perfect
        thresholds for example).

        Parameters
        ----------
        scale : float
            Rescales footage down (to speed up processing).

        window : int
            Number of frames that are also considered (starts with first frame where stopped vehicle is detected).
            Compensates for small movements and noise in footage that could otherwise cause one vehicle to yield
            multiple images.

            The higher the framerate, the longer window is recommended.

        Returns
        -------
        (list, list, list, list)
            Tuple of four datasets, vehicle presence, vehicle motion, vehicle blurriness and window status. All four
            contain data-points for each frame of footage, so even the vehicle blurriness dataset contains data of
            frames where no vehicle is present.

        Raises
        ------
        ValueError
            If path to video or folder doesn't exist or cannot be read.

        RuntimeError
            If failed to fetch first frame from footage.
        """

        if not 0 < scale <= 1:
            raise ValueError('Invalid scale factor (must be between 1 and 0).')

        last_hit = window

        presence = []
        motion = []
        focus = []
        window_status = []

        # Acquire first frame
        prev_frame = self.frame_extractor.next_frame()
        if prev_frame is None:
            raise RuntimeError('Failed fetching first frame from input. Possibly reached the end.')

        # Image processing context
        height, width = prev_frame.shape
        height, width = int(height * scale), int(width * scale)
        blur_kernel = self._calc_kernel_size(width, height)

        # Scale image down
        prev_frame_pp = self._prep_image(prev_frame, scale, blur_kernel)

        curr_frame = self.frame_extractor.next_frame()
        while curr_frame is not None:
            # Process next frame
            curr_frame_pp = self._prep_image(curr_frame, scale, blur_kernel)

            vehicle_presence = np.count_nonzero(self.vehicle_mask(curr_frame_pp)) / (width * height)
            vehicle_motion = np.count_nonzero(self.motion_mask(curr_frame_pp, prev_frame_pp)) / (width * height)

            # Vehicle present and not moving
            if vehicle_presence > self.vehicle_threshold and vehicle_motion < self.motion_threshold:
                # Vehicle focus is variance of Laplacian operator over dark areas of image
                lap_var = self.cached_laplacian_var(curr_frame)
                last_hit = 0
            elif last_hit < window:
                # No stopped car = zero focus
                lap_var = 0
                last_hit += 1
            else:
                # No stopped car = zero focus
                lap_var = 0

            presence.append(vehicle_presence)
            motion.append(vehicle_motion)
            focus.append(lap_var)
            window_status.append(1 if last_hit < window else 0)

            # Next frame
            prev_frame_pp = curr_frame_pp
            curr_frame = self.frame_extractor.next_frame()

        return presence, motion, focus, window_status
