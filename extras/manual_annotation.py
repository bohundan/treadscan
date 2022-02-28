"""
This module can be run as a script from a command line, for arguments, see --help.
"""

from argparse import ArgumentParser
from datetime import datetime

import cv2
from annotator import Annotator
from treadscan.extractor import Extractor, CameraPosition


def main(filename: str, max_width: int, max_height: int, flipped: bool):
    """
    Main function when script is started as a shell program. If annotation is submitted (enter), the extracted tread
    is saved to disk.

    Parameters
    ----------
    filename : str
        Path to image which to annotate. Taken as a command line argument.

    max_width : int
        Max width of annotation window.

    max_height : int
        Max height of annotation window.

    flipped : bool
        If true, image will be flipped horizontally.
    """

    if max_width < 100 or max_height < 100:
        print('Maximum width or height is too small.')
        return

    # Annotate image
    original_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    annotator = Annotator(original_image, max_width, max_height, flipped)
    result = annotator.annotate()

    # If user submitted annotation
    if result is not None:
        # Get parameters
        main_ellipse, outer_ellipse, inner_ellipse, tread_width, start_angle, end_angle, flipped = result

        if flipped:
            position = CameraPosition.FRONT_LEFT
        else:
            position = CameraPosition.FRONT_RIGHT

        # Extract tread
        extractor = Extractor(original_image, main_ellipse, position=position)
        tread = extractor.extract_tread(tire_width=tread_width, tire_bounding_ellipses=(outer_ellipse, inner_ellipse),
                                        start=start_angle, end=end_angle, cores=2)

        # Write to file
        filename = f'tread_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png'
        cv2.imwrite(filename, tread)
        print(f'Tread saved as \'{filename}\'.')

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
args = arg_parser.parse_args()

if __name__ == '__main__':
    main(args.filename, args.width, args.height, args.flipped)