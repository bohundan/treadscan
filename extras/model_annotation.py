"""
This module can be run as a script from a command line, for arguments, see --help.
"""

from argparse import ArgumentParser
import os
from pathlib import Path

import cv2
from annotator import Annotator


def main(filename: str, max_width: int, max_height: int, rescale: float):
    """
    Main function when script is started as a shell program. If annotation is submitted (enter), the keypoints are saved
    as json file.

    Parameters
    ----------
    filename : str
        Path to image which to annotate or path to json file with keypoints

    max_width : int
        Max width of annotation window.

    max_height : int
        Max height of annotation window.

    rescale : float
        If json file provided as filename and rescale is not 1, keypoints in json file will be rescaled.
    """

    path = Path(filename)

    # Rescale keypoints in json file
    if path.suffix == '.json':
        if rescale != 1:
            print('KEYPOINT RESCALING NOT IMPLEMENTED')
        else:
            print('You need to provide an image (without .json suffix) or rescale keypoints by different factor than 1')
    # Else annotate keypoints and write to json file
    else:
        if max_width < 100 or max_height < 100:
            print('Maximum width or height is too small.')
            return

        if path.is_dir():
            filenames = [str(f) for f in path.iterdir() if f.is_file()]
            filenames.sort()
        else:
            filenames = [filename]

        for filename in filenames:
            # Annotate images
            original_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            annotator = Annotator(original_image, max_width, max_height)
            result = annotator.annotate_keypoints()

            # If user submitted annotation
            if result:
                images_folder = 'images_'
                annotations_folder = 'annotations_'

                Path(images_folder).mkdir(parents=True, exist_ok=True)
                Path(annotations_folder).mkdir(parents=True, exist_ok=True)

                stem = Path(filename).stem
                # Write image to images folder
                cv2.imwrite(f'{images_folder}/{stem}.jpg', original_image)
                # Write annotation to annotations folder
                with open(f'{annotations_folder}/{stem}.json', 'w') as json_file:
                    json_file.write(result)


def parse_filename(parser: ArgumentParser, arg: str):
    """
    Checks existence of filename argument.

    Parameters
    ----------
    parser : ArgumentParser

    arg : str
        Filename to check.

    Returns
    -------
    None
        If file or folder doesn't exist.

    str
        Filename if file or folder exists and is readable.
    """

    path = Path(arg)

    # Directory
    if path.is_dir():
        files = [str(f) for f in path.iterdir() if f.is_file()]

        if len(files) == 0:
            parser.error(f'Folder {arg} is empty!')
            return
        try:
            print(files[0])
            file = open(files[0], 'r')
        except IOError:
            parser.error(f'Files in {arg} are not readable!')
            return

        file.close()
        return arg

    # One file
    else:
        try:
            file = open(arg, 'r')
        except IOError:
            parser.error(f'The file {arg} does not exist!')
            return

        file.close()
        return arg

arg_parser = ArgumentParser(description='Manually create tire bounding ellipses for tread extraction.')
arg_parser.add_argument('-i', dest='filename', required=True, help='input image for annotation', metavar='FILE/FOLDER',
                        type=lambda x: parse_filename(arg_parser, x))
arg_parser.add_argument('--width', dest='width', required=False, help='maximum width of annotation window',
                        metavar='NUMBER', type=int, default=1800)
arg_parser.add_argument('--height', dest='height', required=False, help='maximum height of annotation window',
                        metavar='NUMBER', type=int, default=800)
arg_parser.add_argument('--rescale', dest='rescale', required=False, help='rescale keypoints inside file by factor',
                        metavar='NUMBER', type=float, default=1.0)
args = arg_parser.parse_args()

if __name__ == '__main__':
    main(args.filename, args.width, args.height, args.rescale)
