"""
This module can be run as a script from a command line, for arguments, see --help.
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
from annotator import Annotator
from treadscan.extractor import CameraPosition


def main(filename: str, max_width: int, max_height: int, flipped: bool, rescale: float):
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

    flipped : bool
        If true, image will be flipped horizontally.

    rescale : float
        If json file provided as filename and rescale is not 1, keypoints in json file will be rescaled.
    """

    path = Path(filename)

    # Rescale keypoints in json file
    if path.suffix == '.json':
        if rescale != 1:
            print('NOT IMPLEMENTED')
        else:
            print('You need to provide an image (without .json suffix) or rescale keypoints by different factor than 1')
    # Else annotate keypoints and write to json file
    else:
        if max_width < 100 or max_height < 100:
            print('Maximum width or height is too small.')
            return

        # Annotate image
        original_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        annotator = Annotator(original_image, max_width, max_height, flipped)
        result = annotator.annotate_keypoints()

        # If user submitted annotation
        if result:
            if flipped:
                image = cv2.flip(original_image, 1)
            else:
                image = original_image

            Path('images').mkdir(parents=True, exist_ok=True)
            Path('annotations').mkdir(parents=True, exist_ok=True)

            stem = path.stem
            # Write image to images folder
            cv2.imwrite(f'images/{stem}.jpg', image)
            # Write annotation to annotations folder
            with open(f'annotations/{stem}.json', 'w') as json_file:
                json_file.write(result)


def parse_filename(parser: ArgumentParser, arg: str):
    """
    Checks existence of filename argument.

    Parameters
    ----------
    parser : argparseArgumentParser

    arg : str
        Filename to check.

    Returns
    -------
    None
        If file doesn't exist.

    str
        Filename if file exists and is readable.
    """

    try:
        file = open(arg, 'r')
    except IOError:
        parser.error(f'The file {arg} does not exist!')
        return

    file.close()
    return arg

arg_parser = ArgumentParser(description='Manually create tire bounding ellipses for tread extraction.')
arg_parser.add_argument('-i', dest='filename', required=True, help='input image for annotation', metavar='FILE',
                        type=lambda x: parse_filename(arg_parser, x))
arg_parser.add_argument('--width', dest='width', required=False, help='maximum width of annotation window',
                        metavar='NUMBER', type=int, default=1800)
arg_parser.add_argument('--height', dest='height', required=False, help='maximum height of annotation window',
                        metavar='NUMBER', type=int, default=800)
arg_parser.add_argument('--flipped', dest='flipped', required=False, help='image will be flipped horizontally',
                        action='store_true')
arg_parser.add_argument('--rescale', dest='rescale', required=False, help='rescale keypoints inside file by factor',
                        metavar='NUMBER', type=float, default=1.0)
args = arg_parser.parse_args()

if __name__ == '__main__':
    main(args.filename, args.width, args.height, args.flipped, args.rescale)
