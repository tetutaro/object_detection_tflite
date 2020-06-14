#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Generator, Optional
import os
import io
import time
import click
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import numpy as np
import tflite_runtime.interpreter as tflite
import platform
if platform.system() == 'Linux':  # RaspberryPi
    import picamera
    EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
    DEFAULT_HFLIP = True
    DEFAULT_VFLIP = True
elif platform.system() == 'Darwin':  # MacOS X
    import cv2
    EDGETPU_SHARED_LIB = 'libedgetpu.1.dylib'
    DEFAULT_HFLIP = False
    DEFAULT_VFLIP = False
else:
    raise NotImplementedError()

FRAME_PER_SECOND = 30


def _round_up(value: int, n: int) -> int:
    return n * ((value + (n - 1)) // n)


def _round_buffer_dims(dims: Tuple[int, int]) -> Tuple[int, int]:
    width, height = dims
    return _round_up(width, 32), _round_up(height, 16)


class Camera(object):
    def __init__(
        self: Camera,
        width: int,
        height: int,
        threshold: float,
        fontsize: int
    ) -> None:
        self._dims = (width, height)
        self._buffer = Image.new('RGBA', self._dims)
        self._overlay = None
        self._draw = ImageDraw.Draw(self._buffer)
        self._default_color = (0xff, 0xff, 0xff, 0xff)
        self._font = ImageFont.truetype(
            font='TakaoGothic.ttf', size=fontsize
        )
        self._threshold = threshold
        self._fontsize = fontsize
        return

    def clear(self: Camera) -> None:
        self._draw.rectangle(
            (0, 0) + self._dims,
            fill=(0, 0, 0, 0x00)
        )
        return

    def draw_objects(self: Camera, objects: List) -> None:
        for obj in objects:
            self._draw_object(obj)
        return

    def draw_time(self: Camera, elapsed_ms: float) -> None:
        self._draw_text(
            'Elapsed Time: %.1f[ms]' % elapsed_ms,
            location=(5, 5), color=None
        )
        return

    def draw_count(self: Camera, count: int) -> None:
        self._draw_text(
            'Detected Objects: %d' % count,
            location=(5, 5 + self._fontsize), color=None
        )
        return

    def _draw_object(self: Camera, object: Dict) -> None:
        prob = object['prob']
        color = tuple(np.array(np.array(cm.jet(
            (prob - self._threshold) / (1.0 - self._threshold)
        )) * 255, dtype=np.uint8).tolist())
        self._draw_box(
            rect=object['bbox'], color=color
        )
        name = object.get('name')
        xoff = object['bbox'][0] + 5
        yoff = object['bbox'][1] + 5
        if name is not None:
            self._draw_text(
                name, location=(xoff, yoff), color=color
            )
            yoff += self._fontsize
        self._draw_text(
            '%.3f' % prob, location=(xoff, yoff), color=color
        )
        return

    def _draw_box(
        self: Camera,
        rect: Tuple[int, int, int, int],
        color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        outline = color or self._default_color
        self._draw.rectangle(rect, fill=None, outline=outline)
        return

    def _draw_text(
        self: Camera,
        text: str,
        location: Tuple[int, int],
        color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        color = color or self._default_color
        self._draw.text(location, text, fill=color, font=self._font)
        return

    def update(self: Camera) -> None:
        if self._overlay is not None:
            self._camera.remove_overlay(self._overlay)
        if self._buffer is None:
            return
        self._overlay = self._camera.add_overlay(
            self._buffer.tobytes(),
            format='rgba', layer=3, size=self._dims
        )
        self._overlay.update(self._buffer.tobytes())
        return


class RaspberryPiCamera(Camera):
    def __init__(
        self: RaspberryPiCamera,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        threshold: float,
        fontsize: int
    ) -> None:
        super().__init__(
            width=width, height=height,
            threshold=threshold, fontsize=fontsize
        )
        self._camera = picamera.PiCamera(
            resolution=(width, height),
            framerate=FRAME_PER_SECOND
        )
        self._camera.hflip = hflip
        self._camera.vflip = vflip
        return

    def start(self: RaspberryPiCamera) -> None:
        self._camera.start_preview()
        return

    def yield_image(self: RaspberryPiCamera) -> Generator[Image, None]:
        self._stream = io.BytesIO()
        for _ in self._camera.capture_continuous(
            self._stream,
            format='jpeg',
            use_video_port=True
        ):
            self._stream.seek(0)
            image = Image.open(self._stream).convert('RGB')
            yield image
        return

    def update(self: RaspberryPiCamera) -> None:
        super().update()
        self._stream.seek(0)
        self._stream.truncate()
        return

    def stop(self: RaspberryPiCamera) -> None:
        self._camera.stop_preview()
        return


class MacCamera(Camera):
    def __init__(
        self: MacCamera,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        threshold: float,
        fontsize: int
    ) -> None:
        super().__init__(
            width=width, height=height,
            threshold=threshold, fontsize=fontsize
        )
        self._camera = cv2.VideoCapture(0)
        # adjust aspect ratio
        video_width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if video_width == width and video_height == height:
            self.resize_frame = None
            self.crop_frame = None
        else:
            width_scale = float(width) / video_width
            height_scale = float(height) / video_height
            scale = max(width_scale, height_scale)
            resize_width = int(np.ceil(scale * video_width))
            resize_height = int(np.ceil(scale * video_height))
            if scale == 1.0:
                self.resize_frame = None
            else:
                self.resize_frame = (resize_width, resize_height)
            width_margin = int((resize_width - width) * 0.5)
            height_margin = int((resize_height - height) * 0.5)
            self.crop_frame = (
                width_margin, width_margin + width,
                height_margin, height_margin + height
            )
        # set flipcode
        if hflip:
            if vflip:
                self.flipcode = -1
            else:
                self.flipcode = 1
        elif vflip:
            self.flipcode = 0
        else:
            self.flipcode = None
        return

    def start(self: MacCamera) -> None:
        self.window = 'Object Detection'
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window, *self._dims)
        return

    def yield_image(self: MacCamera) -> Generator[Image, None]:
        while True:
            _, image = self._camera.read()
            if image is None:
                time.sleep(1)
                continue
            if self.resize_frame is not None:
                image = cv2.resize(image, self.resize_frame)
            if self.crop_frame is not None:
                xmin, xmax, ymin, ymax = self.crop_frame
                image = image[ymin:ymax, xmin:xmax]
            if self.flipcode is not None:
                image = cv2.flip(image, self.flipcode)
            self.image = image
            yield Image.fromarray(image.copy()[..., ::-1])
        return

    def update(self: MacCamera) -> None:
        overlay = np.array(self._buffer, dtype=np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        image = cv2.addWeighted(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
            overlay, 0.5, 2.2
        )
        cv2.imshow(self.window, image)
        key = cv2.waitKey(1000 // FRAME_PER_SECOND)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: MacCamera) -> None:
        cv2.destroyAllWindows()
        self._camera.release()
        return


def get_camera(
    width: int,
    height: int,
    hflip: bool,
    vflip: bool,
    threshold: float,
    fontsize: int
) -> Camera:
    if platform.system() == 'Linux':  # RaspberryPi
        camera = RaspberryPiCamera(
            width=width, height=height,
            hflip=hflip, vflip=vflip,
            threshold=threshold, fontsize=fontsize
        )
    elif platform.system() == 'Darwin':  # MacOS
        camera = MacCamera(
            width=width, height=height,
            hflip=hflip, vflip=vflip,
            threshold=threshold, fontsize=fontsize
        )
    else:
        raise NotImplementedError()
    return camera


class Detector(object):
    def __init__(
        self: Detector,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        model_path: str,
        tpu: bool,
        target: str,
        threshold: float,
        fontsize: int
    ) -> None:
        # load label
        if not os.path.exists('models/coco_labels.txt'):
            raise Exception("do download_modes.sh")
        self.label2id = dict()
        self.id2label = dict()
        with open('models/coco_labels.txt', 'rt', encoding='utf-8') as rf:
            line = rf.readline()
            while line:
                info = [x.strip() for x in line.strip().split(' ', 1)]
                idx = int(info[0])
                label = info[1].replace(' ', '_')
                self.id2label[idx] = label
                self.label2id[label] = idx
                line = rf.readline()
        # set target
        if target == 'all':
            self.target_id = -1
        else:
            if target not in self.label2id.keys():
                raise ValueError('target not in models/coco_labels.txt')
            self.target_id = self.label2id[target]
        # load interpreter
        if tpu:
            delegates = [
                tflite.load_delegate(EDGETPU_SHARED_LIB, {})
            ]
        else:
            delegates = None
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()
        # get input and output details
        input_detail = self.interpreter.get_input_details()[0]
        self.input_index = input_detail['index']
        shape = input_detail['shape']
        self.image_height = shape[1]
        self.image_width = shape[2]
        output_details = self.interpreter.get_output_details()
        self.boxes_index = output_details[0]['index']
        self.class_index = output_details[1]['index']
        self.score_index = output_details[2]['index']
        self.count_index = output_details[3]['index']
        # set paramters
        self.camera_width = width
        self.camera_height = height
        self.hflip = hflip
        self.vflip = vflip
        self.threshold = threshold
        self.fontsize = fontsize
        return

    def detect_objects(self: Detector, image: Image) -> Tuple[List, float]:
        # set input
        image = image.resize(
            (self.image_width, self.image_height),
            Image.ANTIALIAS
        )
        self.interpreter.set_tensor(
            self.input_index,
            np.array(image)[np.newaxis, ...]
        )
        # invoke
        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        # get output
        boxes = self.interpreter.get_tensor(self.boxes_index).squeeze()
        class_ids = self.interpreter.get_tensor(self.class_index).squeeze()
        scores = self.interpreter.get_tensor(self.score_index).squeeze()
        count = int(self.interpreter.get_tensor(self.count_index).squeeze())
        results = list()
        for i in range(count):
            cid = int(class_ids[i])
            prob = float(scores[i])
            name = self.id2label.get(cid)
            if name is None:
                continue
            if self.target_id != -1 and self.target_id != cid:
                continue
            if prob < self.threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = max(0, int(ymin * self.camera_height))
            xmin = max(0, int(xmin * self.camera_width))
            ymax = min(self.camera_height, int(ymax * self.camera_height))
            xmax = min(self.camera_width, int(xmax * self.camera_width))
            results.append({
                'name': name,
                'prob': prob,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        return results, elapsed_ms

    def run(self: Detector) -> None:
        camera = get_camera(
            width=self.camera_width,
            height=self.camera_height,
            hflip=self.hflip,
            vflip=self.vflip,
            threshold=self.threshold,
            fontsize=self.fontsize
        )
        camera.start()
        try:
            for image in camera.yield_image():
                objects, elapsed_ms = self.detect_objects(image)
                camera.clear()
                camera.draw_objects(objects)
                camera.draw_time(elapsed_ms)
                camera.draw_count(len(objects))
                camera.update()
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
        return


@click.command()
@click.option('--width', type=int, default=640)
@click.option('--height', type=int, default=640)
@click.option('--hflip/--no-hflip', is_flag=True, default=DEFAULT_HFLIP)
@click.option('--vflip/--no-vflip', is_flag=True, default=DEFAULT_VFLIP)
@click.option('--tpu/--no-tpu', is_flag=True, default=False)
@click.option('--model', type=str, default='coco')
@click.option('--target', type=str, default='all')
@click.option('--threshold', type=float, default=0.5)
@click.option('--fontsize', type=int, default=20)
def main(
    width: int,
    height: int,
    hflip: bool,
    vflip: bool,
    tpu: bool,
    model: str,
    target: str,
    threshold: float,
    fontsize: int
) -> None:
    model_path = 'models/mobilenet_ssd_v2_'
    if model == 'face':
        if tpu:
            model_path += 'face_quant_postprocess_edgetpu.tflite'
        else:
            model_path += 'face_quant_postprocess.tflite'
    else:
        if tpu:
            model_path += 'coco_quant_postprocess_edgetpu.tflite'
        else:
            model_path += 'coco_quant_postprocess.tflite'
    assert(os.path.exists(model_path))
    width, height = _round_buffer_dims((width, height))
    detector = Detector(
        width=width,
        height=height,
        hflip=hflip,
        vflip=vflip,
        model_path=model_path,
        tpu=tpu,
        target=target,
        threshold=threshold,
        fontsize=fontsize
    )
    detector.run()
    return


if __name__ == "__main__":
    main()
