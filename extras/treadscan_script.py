"""
This module can be run as a script from a command line, for arguments, see --help.
"""

from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from os.path import isdir
from pathlib import Path

import cv2
from annotator import Annotator
from treadscan.detector import Detector, InputType
from treadscan.segmentor import Segmentor
from treadscan.extractor import Extractor, CameraPosition


class Mode(Enum):
    """
    Enum class defining modes of operation of this script.

    MANUAL :
        Annotate each image manually.

    SEMIAUTO :
        Try to find ellipse automatically, if fails, annotate image manually.

    AUTO :
        Find ellipse automatically, if fails, discard.
    """

    MANUAL = 1
    SEMIAUTO = 2
    AUTO = 3


def main(input_path: str, background_path: str, output_folder: str, flipped: bool, mode: Mode):
    """
    Main function when script is started as a shell program.

    Parameters
    ----------
    input_path : str
        Path to input file or folder.

    background_path : str
        Path to background sample (image).

    output_folder : str
        Path to folder in which to save extracted treads. If it does not exist, it will be created.

    flipped : bool
        If images are to be flipped horizontally (mirrored). Refer to treadscan.extractor.CameraPosition.

    mode : Mode
        Automatic, semiautomatic or manual ellipse detection.
    """

    # Detect type of input
    if isdir(input_path):
        input_type = InputType.IMAGE_FOLDER
    else:
        input_type = InputType.VIDEO

    # Create output folder if specified
    if output_folder:
        if output_folder[-1] == '/':
            output_folder = output_folder[:-1]
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Create detector
    try:
        detector = Detector(background_path)
    except ValueError as error:
        print(error)
        return

    if flipped:
        position = CameraPosition.FRONT_LEFT
    else:
        position = CameraPosition.FRONT_RIGHT

    gen = detector.detect_background_subtraction(input_path, input_type)
    while True:
        try:
            image = next(gen)
        except StopIteration:
            print('End of footage reached.')
            break
        except ValueError as error:
            print(f'Encountered error: \'{error}\'.')
            break

        main_ellipse = None
        current_position = position
        kwargs = None

        if mode == Mode.AUTO or mode == Mode.SEMIAUTO:
            # Try to detect ellipse
            segmentor = Segmentor(image)
            main_ellipse = segmentor.find_ellipse()
            current_position = position
            kwargs = {}

        if mode == Mode.MANUAL or (mode == Mode.SEMIAUTO and main_ellipse is None):
            # Manual ellipse detection
            annotator = Annotator(image, 1800, 800, flipped)
            result = annotator.annotate()
            if result is not None:
                main_ellipse, outer_ellipse, inner_ellipse, tread_width, start_angle, end_angle, flipped = result

                if flipped:
                    current_position = CameraPosition.FRONT_LEFT
                else:
                    current_position = CameraPosition.FRONT_RIGHT
                kwargs = {
                    'tire_width': tread_width,
                    'tire_bounding_ellipses': (outer_ellipse, inner_ellipse),
                    'start': start_angle,
                    'end': end_angle
                }

        if main_ellipse is not None:
            # Ellipse found, extract tread
            extractor = Extractor(image, main_ellipse, position=current_position)
            try:
                tread = extractor.extract_tread(**kwargs)
            except ValueError as error:
                print(f'Failed extracting tread: \'{error}\'')
                continue

            cv2.imwrite(f'{output_folder}/tread_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}.png', tread)


arg_parser = ArgumentParser(description='Extract tire treads from each stopped car in footage.')
arg_parser.add_argument('-b', dest='background', required=True, help='path to sample of background (image)',
                        metavar='FILE', type=str)
arg_parser.add_argument('-i', dest='input', required=True, help='path to video or folder to scan from',
                        metavar='FILE or FOLDER', type=str)
arg_parser.add_argument('-o', dest='folder', required=False, help='folder in which to save scanned treads',
                        metavar='FOLDER', type=str, default='')
arg_parser.add_argument('--flipped', dest='flipped', required=False,
                        help='use this flag if tire tread is on the left side relative to visible wheel, refer to treadscan.extractor.CameraPosition',
                        action='store_true')
arg_parser.add_argument('--manual', dest='manual', required=False, help='manual mode', action='store_true')
arg_parser.add_argument('--semiauto', dest='semiauto', required=False, help='semi automatic mode', action='store_true')
arg_parser.add_argument('--auto', dest='auto', required=False, help='automatic mode', action='store_true')
args = arg_parser.parse_args()

if __name__ == '__main__':
    if args.manual:
        main(args.input, args.background, args.folder, args.flipped, Mode.MANUAL)
    elif args.semiauto:
        main(args.input, args.background, args.folder, args.flipped, Mode.SEMIAUTO)
    elif args.auto:
        main(args.input, args.background, args.folder, args.flipped, Mode.AUTO)
    else:
        main(args.input, args.background, args.folder, args.flipped, Mode.AUTO)
