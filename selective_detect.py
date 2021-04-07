#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import platform
if platform.system() == 'Linux':  # RaspberryPi
    DEFAULT_HFLIP = True
    DEFAULT_VFLIP = True
elif platform.system() == 'Darwin':  # MacOS X
    DEFAULT_HFLIP = False
    DEFAULT_VFLIP = False
else:
    raise NotImplementedError()
from src.config import Config
from src.detector import DetectorSelective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="selective search + image classification"
    )
    parser.add_argument(
        '--media', type=str, default=None,
        help=(
            'filename of image/video'
            ' (if not set, use streaming video from camera)'
        )
    )
    parser.add_argument(
        '--height', type=int, default=720,
        help='camera image height'
    )
    parser.add_argument(
        '--width', type=int, default=1280,
        help='camera image width'
    )
    parser.add_argument(
        '--hflip',
        action='store_false' if DEFAULT_HFLIP else 'store_true',
        help='flip horizontally'
    )
    parser.add_argument(
        '--vflip',
        action='store_false' if DEFAULT_VFLIP else 'store_true',
        help='flip vertically'
    )
    parser.add_argument(
        '--model', type=str, default='mobilenet',
        choices=['mobilenet', 'bird', 'insect', 'plant'],
        help='object detection model'
    )
    parser.add_argument(
        '--quant', type=str, default='fp32',
        choices=['fp32', 'tpu'],
        help='quantization mode (or use EdgeTPU)'
    )
    parser.add_argument(
        '--target', type=str, default='all',
        help='the target type of detecting object (default: all)'
    )
    parser.add_argument(
        '--prob-threshold', type=float, default=0.5,
        help='the hreshold of probability'
    )
    parser.add_argument(
        '--iou-threshold', type=float, default=0.45,
        help='the IoU threshold of NMS'
    )
    parser.add_argument(
        '--fontsize', type=int, default=20,
        help='fontsize to display'
    )
    parser.add_argument(
        '--fastforward', type=int, default=1,
        help=(
            'frame interval for object detection'
            ' (default: 1 = detect every frame)'
        )
    )
    args = parser.parse_args()
    config = Config(**vars(args))
    detector = DetectorSelective(config=config)
    detector.run()
    return


if __name__ == '__main__':
    main()
