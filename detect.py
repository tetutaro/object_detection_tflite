#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional
import os
import time
import click
import platform
if platform.system() == 'Linux':  # RaspberryPi
    DEFAULT_HFLIP = True
    DEFAULT_VFLIP = True
elif platform.system() == 'Darwin':  # MacOS X
    DEFAULT_HFLIP = False
    DEFAULT_VFLIP = False
else:
    raise NotImplementedError()
from camera import get_camera
from decode import get_decoder


def _round_up(value: int, n: int) -> int:
    return n * ((value + (n - 1)) // n)


def _round_buffer_dims(dims: Tuple[int, int]) -> Tuple[int, int]:
    width, height = dims
    return _round_up(width, 32), _round_up(height, 16)


class Detector(object):
    def __init__(
        self: Detector,
        media: Optional[str],
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        model_path: str,
        target: str,
        threshold: float,
        fontsize: int,
        fastforward: int
    ) -> None:
        if fastforward > 1:
            self.fastforward = fastforward
        else:
            self.fastforward = 1
        self.camera = get_camera(
            media=media,
            width=width,
            height=height,
            hflip=hflip,
            vflip=vflip,
            threshold=threshold,
            fontsize=fontsize,
            fastforward=self.fastforward
        )
        width, height = self.camera._dims
        self.decoder = get_decoder(
            model_path=model_path,
            target=target,
            threshold=threshold,
            width=width,
            height=height
        )
        return

    def run(self: Detector) -> None:
        self.camera.start()
        try:
            framecount = 0
            for image in self.camera.yield_image():
                framecount += 1
                if framecount >= self.fastforward:
                    start_time = time.perf_counter()
                    objects = self.decoder.detect_objects(image)
                    objects = self.decoder.get_bboxes(objects)
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    self.camera.clear()
                    self.camera.draw_objects(objects)
                    self.camera.draw_time(elapsed_ms)
                    self.camera.draw_count(len(objects))
                    self.camera.update()
                    framecount = 0
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
        return


@click.command()
@click.option('--media', type=str, default=None)
@click.option('--width', type=int, default=1280)
@click.option('--height', type=int, default=720)
@click.option('--hflip/--no-hflip', is_flag=True, default=DEFAULT_HFLIP)
@click.option('--vflip/--no-vflip', is_flag=True, default=DEFAULT_VFLIP)
@click.option('--tpu/--no-tpu', is_flag=True, default=False)
@click.option('--model', type=str, default='coco')
@click.option('--target', type=str, default='all')
@click.option('--threshold', type=float, default=0.5)
@click.option('--fontsize', type=int, default=20)
@click.option('--fastforward', type=int, default=1)
def main(
    media: Optional[str],
    width: int,
    height: int,
    hflip: bool,
    vflip: bool,
    tpu: bool,
    model: str,
    target: str,
    threshold: float,
    fontsize: int,
    fastforward: int
) -> None:
    if model == 'yolov3':
        model_path = 'yolo/yolov3'
        if tpu:
            model_path += '_edgetpu.tflite'
        else:
            model_path += '.tflite'
    elif model == 'yolov3-tiny':
        model_path = 'yolo/yolov3-tiny'
        if tpu:
            model_path += '_edgetpu.tflite'
        else:
            model_path += '.tflite'
    elif model == 'yolov4':
        model_path = 'yolo/yolov4'
        if tpu:
            model_path += '_edgetpu.tflite'
        else:
            model_path += '.tflite'
    elif model == 'face':
        model_path = 'models/mobilenet_ssd_v2_'
        if tpu:
            model_path += 'face_quant_postprocess_edgetpu.tflite'
        else:
            model_path += 'face_quant_postprocess.tflite'
    else:
        model_path = 'models/mobilenet_ssd_v2_'
        if tpu:
            model_path += 'coco_quant_postprocess_edgetpu.tflite'
        else:
            model_path += 'coco_quant_postprocess.tflite'
    assert(os.path.exists(model_path))
    width, height = _round_buffer_dims((width, height))
    detector = Detector(
        media=media,
        width=width,
        height=height,
        hflip=hflip,
        vflip=vflip,
        model_path=model_path,
        target=target,
        threshold=threshold,
        fontsize=fontsize,
        fastforward=fastforward
    )
    detector.run()
    return


if __name__ == "__main__":
    main()
