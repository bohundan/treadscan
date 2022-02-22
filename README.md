# treadscan

[![PyPI version](https://badge.fury.io/py/treadscan.svg)](https://badge.fury.io/py/treadscan)
[![Documentation Status](https://readthedocs.org/projects/treadscan/badge/?version=latest)](https://treadscan.readthedocs.io/en/latest/?badge=latest)

This package provides utilities for extracting tire tread (pattern) from images, or even from video of passing cars.

![treadscan](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/treadscan.jpg)

Three main classes are  
`Detector` - picks out stopped cars from continoous footage (like at an intersection, under a traffic light).   
`Segmentor` - finds ellipse defining car wheel in image (using image segmentation techniques).  
`Extractor` - creates a tire model from the ellipse, then "unwraps" the tire tread (or rather part of the tire tread, as only about a quarter of the tread is visible).

Currently `Segmentor` works correctly only if the car has bright wheels (rims).  But there is a way to also extract tire tread from images where this detection method fails, by manual input. In the `extras` folder, there is another module `annotator.py` containing the `Annotator` class and the script `manual_annotation.py` which uses it. The script `manual_annotation.py` takes an image as input, which it then allows the user to manually annotate.  
The user places specific points over the image, which in turn creates the car wheel and tire model, from which the tire tread is then extracted as usual.

![manual annotation](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/manual_annotation.png)

This GUI utilises the OpenCV window, using mouse position and keyboard input to place the points. So to place a point, user can hover over a location and press a specific key, or hold the key while dragging with the mouse.  
The points which have to be placed manually are:
- `T` - top of car wheel (not the tire but the rim)
- `B` - bottom of car wheel
- `R` - right (or left) side

These 3 points form an ellipse, which defines the location of car wheel in image (red ellipse in the above image). Then the tire model is extrapolated from that, or can be further modified using the optional points:
- `O` - move outer tire ellipse (left yellow ellipse)
- `S` - tire sidewall height (height of yellow ellipses)
- `W` - tire width (move inner tire ellipse, right yellow ellipse)
- `U` - top of tread extraction (starting angle, top of green rectangle)
- `I` - bottom of tread extraction (ending angle, bottom of green rectangle)

Pressing `SPACEBAR` shows preview of extracted tread, `BACKSPACE` hides this preview. Pressing `ENTER` saves full size extracted tread as new image. Closing the window or pressing `ESCAPE` closes the window.

Additionally the `treadscan_script.py` in the `extras` folder can process footage (folder of images/video/stream) and extract the tread of each car from the footage. It has 3 modes: manual (user annotates each manually), semiauto (user annotates only those where Segmentor fails) and auto (images where Segmentor fails are discarded). Extracted treads are saved to a folder.
