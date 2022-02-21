"""
This module is used for detecting stopped car(s) from camera footage.

For example if used on footage of cars stopping in front of traffic lights, Detector.detect() should only yield one
image per one stopped car. This is achieved by comparing every frame against static background image, when there's
a low match between the two, the frame likely contains another object (a car). There would be more sequential frames
containing an object, so the next step is to compare each two consequent frames with each other. If they're nearly
identical it can be assumed that there's no movement (the longer the time passed between two frames the more obvious
the difference will be if there is movement).

But there can be many such consequent frames, as cars might be stopped waiting at the traffic lights for some time, so
to obtain just one image, these frames (containing stopped car) are compared all against each other by measuring their
Laplacian variance, which in short ensures that the best focused frame will be the one detected.
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
                 Yield best frame of stopped car, car now started moving
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
from typing import Generator, Union

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


class Detector:
    """
    Detects presence and motion of a vehicle from footage, yielding 1 image per 1 stopped car.

    Input can either be a folder full of images (has to be sorted) or a video/stream.

    Attributes
    ----------
    background_sample : np.ndarray
        Grayscale image of the scene without a vehicle present (is used when detection vehicle presence).

    lookup_table : OrderedDict
        HashTable with hashes of images and their Laplacian variance. Used when comparing blurriness of frames to avoid
        needlessly calculating the same variance of the same image multiple times.

    lookup_table_max_size : int
        Maximum size lookup_table can grow to before old values start being purged to limit memory usage.

    background_intensity_threshold : int
        Difference between pixels needed to be classified as 'not background'.

    background_threshold : float
        Minimum percentage of 'not background' pixels to evaluate image as containing a vehicle.

    motion_intensity_threshold : int
        Difference between pixels needed to be classified as 'motion' (between 2 subsequent images).

    motion_threshold : float
        Minimum percentage of 'moving' pixels to evaluate 2 subsequent images as having motion.

    Methods
    -------
    detect(input_path: str, input_type: InputType, scale: float, window: int)
        Generator which yields one image per one stopped car. Uses a window, in which it detects car presence and
        whether it is moving or not. Window opens when a car is detected as stopped and closes when car starts moving
        again. The generator then yields the best focused (least blurry) frame from this window.

    vehicle_present_in_image(image: np.ndarray)
        Determines whether vehicle is present in image or not.

    motion_between_images(image1: np.ndarray, image2: np.ndarray)
        Determines if there is movement between the two images.

    cached_laplacian_var(image: np.ndarray)
        Calculates Laplacian variance of image.

    compare_image_focus(image1: np.ndarray, image2: np.ndarray)
        Compares *blurriness* of the two images, return the less blurry one.
    """

    def __init__(self, background_sample: Union[np.ndarray, str]):
        """
        Parameters
        ----------
        background_sample : np.ndarray or str
            Grayscale image of background of the scene the Detector will be used on.

            Either as raw image (two-dimensional np.ndarray of np.uint8) or path to image, which exists and is readable.

        Raises
        ------
        ValueError
            If path to background_sample doesn't exist or isn't readable (when provided as string).

            If path to background_sample cannot be read by cv2.imread (when provided as string).

            If background_sample isn't a grayscale image.

            If background_sample has invalid resolution (zero pixels tall or wide image).
        """

        # Reading from file
        if type(background_sample) is str:
            if isfile(background_sample):
                image = cv2.imread(background_sample, cv2.IMREAD_GRAYSCALE)
            else:
                raise ValueError('File does not exists or is not readable.')
        # Provided as array
        else:
            image = background_sample

        # Checking validity
        if image is None:
            raise ValueError('Background sample cannot be read.')
        if len(image.shape) != 2:
            raise ValueError('Background sample is not grayscale.')
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError('Background sample is zero pixels tall/wide.')

        # Blur to remove some noise
        self.background_sample = cv2.GaussianBlur(image, (5, 5), 0)

        # Difference between pixels needed to be classified as 'not background'
        self.background_intensity_threshold = 50
        # Minimum percentage of 'not background' pixels to evaluate image as containing a vehicle
        self.background_threshold = 0.2

        # Difference between pixels needed to be classified as 'motion' (between 2 subsequent images)
        self.motion_intensity_threshold = 10
        # Minimum percentage of 'moving' pixels to evaluate 2 subsequent images as having motion
        self.motion_threshold = 0.02

        # Background sample used during the detect() method for background subtraction
        self._background_for_processing = None

        # Structure used for caching some results that can be reused to speed up processing time
        self.lookup_table = OrderedDict()
        self.lookup_table_max_size = 1024

    def detect(self, input_path: str, input_type: InputType, scale: float = 0.25, window: int = 50) \
            -> Generator[np.ndarray, None, None]:
        """
        Generator which yields one frame for footage per one stopped car.

        Parameters
        ----------
        input_path : str
            Path to video/stream or to folder containing images (sorted in sequence).

        input_type : InputType

        scale : float
            Rescales footage down (to speed up processing).

        window : int
            Number of frames that are also considered (starts with first frame where stopped car is detected).
            Compensates for small movements and noise in footage that could otherwise cause one car to yield multiple
            images.

            The higher the framerate, the longer window is recommended.

        Raises
        ------
        ValueError
            If path to video or folder doesn't exist or cannot be read.

        RuntimeError
            If failed to fetch first frame from footage.

            If input frames have different resolution than the background sample.

        Yields
        ------
        np.ndarray
            One image per one stopped car from footage.

        """

        if not 0 < scale <= 1:
            raise ValueError('Invalid scale factor (must be between 1 and 0).')

        self._background_for_processing = scale_image(self.background_sample, scale)

        # Parameters scaled with image resolution (after scaling)
        height, width = self._background_for_processing.shape
        blur_size = height // 50
        blur_size += (1 if blur_size % 2 == 0 else 0)
        blur_kernel = (blur_size, blur_size)

        self._background_for_processing = cv2.blur(self._background_for_processing, blur_kernel)

        frame_extractor = FrameExtractor(input_path, input_type)
        last_hit = window
        best_frame = None

        # Acquire first frame
        prev_frame = frame_extractor.next_frame()
        if prev_frame is None:
            raise RuntimeError('Failed fetching first frame from input.')
        if prev_frame.shape != self.background_sample.shape:
            raise RuntimeError('Footage and background sample have mismatched resolution.')
        prev_frame = scale_image(prev_frame, scale)
        prev_frame = cv2.blur(prev_frame, blur_kernel)

        frame = frame_extractor.next_frame()
        while frame is not None:
            # Process next frame
            if frame.shape != self.background_sample.shape:
                raise RuntimeError('Footage and background sample have mismatched resolution.')
            curr_frame = scale_image(frame, scale)
            curr_frame = cv2.blur(curr_frame, blur_kernel)

            if self.vehicle_present_in_image(curr_frame):
                if not self.motion_between_images(curr_frame, prev_frame):
                    # Vehicle is present and not moving
                    last_hit = 0
                elif last_hit < window:
                    # Vehicle is present but moving
                    last_hit += 1
            elif last_hit < window:
                # No vehicle present
                last_hit += 1

            if last_hit < window:
                # Get the best focused frame in window, only consider frames where no movement is detected
                if last_hit == 0:
                    if best_frame is None:
                        best_frame = frame
                    else:
                        best_frame = self.compare_image_focus(frame, best_frame)
            elif best_frame is not None:
                # Window has closed, yield best focused frame (and unset it to wait for next opened window)
                yield best_frame
                best_frame = None

            # Next frame
            prev_frame = curr_frame
            frame = frame_extractor.next_frame()

        # Make sure to yield last frame if footage ends with window open
        if best_frame is not None:
            yield best_frame

    def vehicle_present_in_image(self, image: np.ndarray) -> bool:
        """
        Detect vehicle in image by subtracting background, thresholding and counting percent (mean) of remaining pixels.
        If a vehicle is present, then the amount of remaining pixels will be high.

        Uses self.background_intensity_threshold as the threshold of minimum difference between image and background,
        and self.background_threshold as the threshold of percentage of pixels that are 'not background' to determine
        presence of vehicle. If the amount of 'non-background' pixels is larger than this threshold, then the image
        is evaluated as containing a vehicle.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        bool
            True when vehicle is detected.

            False when vehicle is not detected.
        """

        # Subtract background
        difference = abs(self._background_for_processing.astype(np.intc) - image.astype(np.intc))
        # Where difference between image and background is more than some intensity threshold
        mask = np.greater(difference, self.background_intensity_threshold)
        # As percentage
        result = np.count_nonzero(mask) / (image.shape[0] * image.shape[1])

        return result > self.background_threshold

    def motion_between_images(self, image1: np.ndarray, image2: np.ndarray) -> bool:
        """
        Determine whether there is movement between two images.

        Uses self.motion_intensity_threshold as the threshold of minimum difference of pixel values to evaluate them as
        "having changed" - moving, and self.motion_threshold as the threshold of percentage of pixels that are "moving"
        to evaluate the images as having motion.

        Parameters
        ----------
        image1 : np.ndarray

        image2 : np.ndarray

        Returns
        -------
        bool
            True when there IS movement

            False when there is NO movement.
        """

        # Difference between images
        difference = abs(image2.astype(np.intc) - image1.astype(np.intc)).astype(np.uint8)
        # Where difference is greater than some intensity threshold
        mask = np.greater(difference, self.motion_intensity_threshold)
        # As percentage
        result = np.count_nonzero(mask) / (image1.shape[0] * image1.shape[1])

        return result > self.motion_threshold

    def cached_laplacian_var(self, image: np.ndarray, strategy: str = 'aggressive'):
        """
        Calculates variance of the Laplacian operator on DARK parts of the image. Uses hashtable to cache results.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        strategy : str
            Strategy to employ when clearing hashtable (making space for new results after it has been filled).

            'aggressive' : Clean entire hashtable (default).

            'lazy' : Remove exactly 1 entry (the oldest one).

            Any other value defaults to 'aggressive' strategy.

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
            small = scale_image(image, 0.25)
            mask = np.less(small, 50).astype(np.uint8)
            dark = small * mask
            self.lookup_table[image_hash] = cv2.Laplacian(dark, cv2.CV_64F).var()

        # Return cached Laplacian variance of image
        return self.lookup_table[image_hash]

    def compare_image_focus(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compare *blurriness* between two images, return the less blurry one.

        Parameters
        ----------
        image1 : np.ndarray

        image2 : np.ndarray

        Returns
        -------
        np.ndarray
            One of the provided images.
        """

        lap_var1 = self.cached_laplacian_var(image1)
        lap_var2 = self.cached_laplacian_var(image2)

        if lap_var1 < lap_var2:
            return image2
        else:
            return image1


class FrameExtractor:
    """
    Class for extracting GRAYSCALE frames from video or folder of images.

    Attributes
    ----------
    input_type : InputType
        Folder of images (has to be sorted alphabetically) or video/stream (cv2.VideoCapture compatible).

    frame_index : int
        Number of frame (frame returned by next_frame), starts at 1 (first frame is frame 1).

    files : list of str
        Sorted list of paths to individual images (when using image folder input).

    video : cv2.VideoCapture
        Loaded video/stream (when using video input).

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

        input_type : InputType

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

    def next_frame(self) -> Union[np.ndarray, None]:
        """
        Gets the next frame from input, converts it to grayscale and returns it.

        Returns
        -------
        np.ndarray
            The next frame as grayscale image.
        None
            When at the end of sequence (end of video/stream).
        """

        # Folder of images
        if self.input_type == InputType.IMAGE_FOLDER:
            if self.frame_index >= len(self.files):
                # At the end
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
