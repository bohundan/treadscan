# treadscan

[![PyPI version](https://badge.fury.io/py/treadscan.svg)](https://badge.fury.io/py/treadscan)
[![Documentation Status](https://readthedocs.org/projects/treadscan/badge/?version=latest)](https://treadscan.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/treadscan)](https://pepy.tech/project/treadscan)

This package provides utilities for extracting tire tread (pattern) from images, or even from video of passing vehicles.

![treadscan](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/treadscan.jpg)

Three main classes are  
`treadscan.Detector` - picks out stopped vehicles from continoous footage (like at an intersection, under a traffic light).
`treadscan.Segmentor` - finds tires in an image, represented by an ellipse (using basic image processing methods) or keypoints (using a Keypoint-RCNN model).
`treadscan.TireModel` - creates a tire model from keypoints (or an ellipse), then "unwraps" the tire tread.

There is a pre-trained Keypoint-RCNN model for detecting tire keypoints available in this repository, or you can train your own. Before training a model, you need annotated training data. To annotate images, you can use the `model_annotation.py`, which conveniently saves the keypoints and bounding boxes as a JSON in the same format as used in the training notebook. This script is included in [this folder](https://github.com/bohundan/treadscan/tree/master/extras).

![model annotation](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/model_annotation.png)

This GUI utilises the OpenCV window, using mouse position and keyboard input to place the points. So to place a point, user can hover over a location and press a specific key, or hold the key while dragging with the mouse.
The points which have to be placed manually are:
- `T` - top of vehicle's rim
- `B` - bottom of vehicle's rim
- `R` - third point on the rim (best to always use the same location)
- `S` - sidewall height (recommended location is above the `T` point)
- `W` - inner side of tire

Other controls are:
- `SPACEBAR` - show a preview of tire tread
- `BACKSPACE` - hide tread preview
- `N` - submit keypoints, keep annotating (you might wish to annotate ALL tires in image if there are multiple)
- `ENTER` - to export keypoints to JSON file
- `ESCAPE` - quit annotation without exporting

The 5 keypoints are used to construct a tire model, defined by outer and inner ellipses.

## Example usage

```python
import cv2
import treadscan

# Grayscale picture of background
background_sample = cv2.imread('background.png', cv2.IMREAD_GRAYSCALE)
# BackgroundSubtractorSimple can be substituted with any cv2 background subtractor, for example cv2.BackgroundSubtractorKNN
background_subtractor = treadscan.BackgroundSubtractorSimple(background_sample)
# Pre-recorded video, could also be a live stream from the camera
frame_extractor = treadscan.FrameExtractor('recording.mp4', treadscan.InputType.VIDEO)

# Detects stopped vehicles from footage
detector = treadscan.Detector(backsub=background_subtractor, frame_extractor=frame_extractor)
# Keypoint-RCNN finds five keypoints per tire in image
segmentor = treadscan.SegmentorRCNN('RCNN_model/saved_model.pth')

i = 1
# For each stopped vehicle from footage:
for image in detector.detect():
    # Save detected vehicle
    cv2.imwrite(f'image{i:04d}.jpg', image)
    # Find all tires in image
    list_of_keypoints = segmentor.find_keypoints(image)
    # For each tire in image:
    for j in range(len(list_of_keypoints)):
        keypoints = list_of_keypoints[j]
        # Construct tire model
        tire_model = treadscan.TireModel(image.shape)
        tire_model.from_keypoints(*keypoints)
        # Unwrap tire tread from the model
        tread = tire_model.unwrap(image)
        # Tread postprocessing
        tread = treadscan.remove_gradient(tread)
        tread = treadscan.clahe(tread)
        # Save unwrapped tread
        cv2.imwrite(f'image{i:04d}_tread{j+1:02d}.jpg', tread)
    i += 1
```
