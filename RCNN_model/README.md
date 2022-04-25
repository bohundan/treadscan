# Training your own vehicle tire detection model with PyTorch

![Tire detection using trained model](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/test_detection.jpg)

### Sources
https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

https://github.com/alexppppp/keypoint_rcnn_training_pytorch

https://github.com/pytorch/vision/tree/main/references/detection

## Preparing your dataset
- You can use provided tools in the `../extras` folder to annotate keypoints for training and testing data. Just run `python3 model_annotation.py -i /path/to/my_image001.jpg`. That opens a cv2 window on which you can use your mouse and keyboard to create keypoints and export them in the required JSON format. For more info, see the `model_annotation.py` and `annotator.py` files.
- Keyboard controls of `model_annotation`:
- `T` - top of vehicle's rim
- `B` - bottom of vehicle's rim
- `R` - third point on the rim (best to always use the same location)
- `S` - sidewall height (recommended location is above the `T` point)
- `W` - inner side of tire
- `SPACEBAR` - show a preview of tire tread
- `BACKSPACE` - hide tread preview
- `N` - submit keypoints, keep annotating (you might wish to annotate ALL tires in image if there are multiple)
- `ENTER` - submit annotation, write image to `images/` folder and JSON annotation to `annotations/` folder
- `ESCAPE` - quit annotation without saving

![model annotation](https://raw.githubusercontent.com/bohundan/treadscan/master/docs/source/_static/model_annotation.png)

## Training the model
- Open the Jupyter notebook `KeypointRCNN_training.ipynb` and run the cells. You may modify any parameters to your liking.
- By default, the model will be saved as `saved_model.pth`.

## Changing the amount of keypoints
- If you wish to add or remove keypoints you can with a few modifications.
- Change the `params.kpt_oks_sigmas` on line 25 in the file `coco_eval.py` to be the same length as the number of keypoints you want to detect.
- Change the `keypoints_transformed_unflattened` on line 43 in cell 3 inside the Jupyter notebook to reshape to the correct dimension. For example for 5 keypoints you have to reshape it to `(-1, 5, 2)`, for 3 keypoints to `(-1, 3, 2)`.
- Change `num_keypoints` parameter from 5 to whatever the number of keypoints you wish whenever you're creating a model using the `get_model()` method inside the Jupyter notebook.

## Changing the amount of classes detected
- You will have to figure this one out yourself as I had no need for it. You might want to look elsewhere if you wish to train your own model to detect more than 1 class of object.
