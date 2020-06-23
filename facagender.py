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

    def draw_faces(self: Camera, objects: List) -> None:
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
        age = object.get('age')
        if age is None:
            age = 'XX'
        else:
            age = '%d' % age
        gender = object.get('gender')
        if gender is None:
            gender = 'X'
        xoff = object['bbox'][0] + 5
        yoff = object['bbox'][1] + 5
        self._draw_text(
            age + '-' + gender, location=(xoff, yoff), color=color
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


class PiCamera(Camera):
    def __init__(
        self: PiCamera,
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

    def start(self: PiCamera) -> None:
        self._camera.start_preview()
        return

    def yield_image(self: PiCamera) -> Generator[Image, None]:
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

    def update(self: PiCamera) -> None:
        super().update()
        self._stream.seek(0)
        self._stream.truncate()
        return

    def stop(self: PiCamera) -> None:
        self._camera.stop_preview()
        return


class CvCamera(Camera):
    def __init__(
        self: CvCamera,
        width: int,
        height: int,
        hflip: bool,
        vflip: bool,
        threshold: float,
        fontsize: int
    ) -> None:
        self._camera = cv2.VideoCapture(0)
        # adjust aspect ratio
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        super().__init__(
            width=width, height=height,
            threshold=threshold, fontsize=fontsize
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

    def start(self: CvCamera) -> None:
        self.window = 'Object Detection'
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window, *self._dims)
        return

    def yield_image(self: CvCamera) -> Generator[Image, None]:
        while True:
            _, image = self._camera.read()
            if image is None:
                time.sleep(1)
                continue
            if self.flipcode is not None:
                image = cv2.flip(image, self.flipcode)
            self.image = image
            yield Image.fromarray(image.copy()[..., ::-1])
        return

    def update(self: CvCamera) -> None:
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

    def stop(self: CvCamera) -> None:
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
        camera = PiCamera(
            width=width, height=height,
            hflip=hflip, vflip=vflip,
            threshold=threshold, fontsize=fontsize
        )
    elif platform.system() == 'Darwin':  # MacOS
        camera = CvCamera(
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
        face_model_path: str,
        agender_model_path: str,
        threshold: float,
        fontsize: int
    ) -> None:
        if 'edgetpu' in face_model_path:
            self.is_tpu = True
        else:
            self.is_tpu = False
        # load label
        if not os.path.exists('models/coco_labels.txt'):
            raise Exception("do download_modes.sh")
        self.label2id = dict()
        self.id2label = dict()
        with open('models/coco_labels.txt', 'rt', encoding='utf-8') as rf:
            off = 0
            line = rf.readline()
            while line:
                info = [x.strip() for x in line.strip().split(' ', 1)]
                idx = int(info[0])
                label = info[1].replace(' ', '_')
                self.id2label[idx] = label
                self.label2id[label] = idx
                line = rf.readline()
                off += 1
        # set target
        if 'person' not in self.label2id.keys():
            raise ValueError('target not in models/coco_labels.txt')
        self.target_id = self.label2id['person']
        # load interpreter
        if self.is_tpu:
            delegates = [
                tflite.load_delegate(EDGETPU_SHARED_LIB, {})
            ]
        else:
            delegates = None
        # face model
        self.face_interpreter = tflite.Interpreter(
            model_path=face_model_path,
            experimental_delegates=delegates
        )
        self.face_interpreter.allocate_tensors()
        # get input and output details
        face_input_detail = self.face_interpreter.get_input_details()[0]
        self.face_input_index = face_input_detail['index']
        shape = face_input_detail['shape']
        self.image_height = shape[1]
        self.image_width = shape[2]
        face_output_details = self.face_interpreter.get_output_details()
        self.face_output_indexes = [
            face_output_details[i]['index'] for i in range(
                len(face_output_details)
            )
        ]
        # agender model
        self.agender_interpreter = tflite.Interpreter(
            model_path=agender_model_path,
            experimental_delegates=delegates
        )
        self.agender_interpreter.allocate_tensors()
        agender_input_detail = self.agender_interpreter.get_input_details()[0]
        self.agender_input_index = agender_input_detail['index']
        agender_output_details = self.agender_interpreter.get_output_details()
        self.agender_output_indexes = [
            agender_output_details[i]['index'] for i in range(
                len(agender_output_details)
            )
        ]
        # set paramters
        self.camera_width = width
        self.camera_height = height
        self.scale_width = float(self.camera_width) / self.image_width
        self.scale_height = float(self.camera_height) / self.image_height
        self.hflip = hflip
        self.vflip = vflip
        self.threshold = threshold
        self.fontsize = fontsize
        return

    def reculc_size(self: Detector, size: Tuple[int]) -> None:
        self.camera_width, self.camera_height = size
        # camera2image
        c2i_scale = min(
            self.image_width / self.camera_width,
            self.image_height / self.camera_height
        )
        self.c2i_size = (
            int(self.camera_width * c2i_scale),
            int(self.camera_height * c2i_scale)
        )
        self.c2i_offset = (
            (self.image_width - self.c2i_size[0]) // 2,
            (self.image_height - self.c2i_size[1]) // 2
        )
        # image2camera
        self.i2c_scale = max(
            self.camera_width / self.image_width,
            self.camera_height / self.image_height
        )
        self.i2c_dummysize = (
            int(self.image_width * self.i2c_scale),
            int(self.image_height * self.i2c_scale)
        )
        self.i2c_offset = (
            (self.i2c_dummysize[0] - self.camera_width) // 2,
            (self.i2c_dummysize[1] - self.camera_height) // 2
        )
        return

    def get_faces(self: Detector, outputs: List) -> List:
        assert(len(outputs) == 4)
        boxes = outputs[0].squeeze()
        class_ids = outputs[1].squeeze()
        scores = outputs[2].squeeze()
        count = int(outputs[3].squeeze())
        frames = list()
        for i in range(count):
            cid = int(class_ids[i])
            prob = float(scores[i])
            name = self.id2label.get(cid)
            if name is None:
                continue
            if self.target_id != cid:
                continue
            if prob < self.threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = max(
                0,
                int(xmin * self.i2c_dummysize[0]) - self.i2c_offset[0]
            )
            ymin = max(
                0,
                int(ymin * self.i2c_dummysize[1]) - self.i2c_offset[1]
            )
            xmax = min(
                self.camera_width,
                int(xmax * self.i2c_dummysize[0]) - self.i2c_offset[0]
            )
            ymax = min(
                self.camera_height,
                int(ymax * self.i2c_dummysize[1]) - self.i2c_offset[1]
            )
            if xmin < xmax and ymin < ymax:
                frames.append({
                    'prob': prob,
                    'bbox': (xmin, ymin, xmax, ymax),
                })
        return frames

    def detect_faces(self: Detector, image: Image) -> List:
        # set input
        image = image.resize(self.c2i_size, Image.ANTIALIAS)
        background = Image.new(
            "RGB", (self.image_width, self.image_height), (128, 128, 128)
        )
        background.paste(image, self.c2i_offset)
        image = np.array(background, dtype=np.uint8)[np.newaxis, ...]
        self.face_interpreter.set_tensor(
            self.face_input_index, image
        )
        # invoke
        self.face_interpreter.invoke()
        # get outputs
        outputs = [
            self.face_interpreter.get_tensor(
                i
            ) for i in self.face_output_indexes
        ]
        # get frames from outputs
        frames = self.get_faces(outputs)
        return frames

    def predict_faces(self: Detector, image: Image, faces: List) -> None:
        for face in faces:
            xmin, ymin, xmax, ymax = face['bbox']
            faceimg = image.crop(
                (xmin, ymin, xmax, ymax)
            ).resize(
                (64, 64), Image.ANTIALIAS
            )
            faceimg = np.array(faceimg, dtype=np.float32)[np.newaxis, ...]
            self.agender_interpreter.set_tensor(
                self.agender_input_index, faceimg
            )
            self.agender_interpreter.invoke()
            outputs = [
                self.agender_interpreter.get_tensor(
                    i
                ) for i in self.agender_output_indexes
            ]
            for output in outputs:
                output = np.squeeze(output)
                if output.shape[0] == 2:
                    # gender
                    if output[0] < 0.5:
                        face['gender'] = 'M'
                    else:
                        face['gender'] = 'F'
                else:
                    # age
                    age = output.dot(np.arange(0, 101).reshape(101, 1))
                    face['age'] = int(age[0])
        return

    def run(self: Detector) -> None:
        camera = get_camera(
            width=self.camera_width,
            height=self.camera_height,
            hflip=self.hflip,
            vflip=self.vflip,
            threshold=self.threshold,
            fontsize=self.fontsize
        )
        self.reculc_size(camera._dims)
        camera.start()
        try:
            for image in camera.yield_image():
                start_time = time.perf_counter()
                faces = self.detect_faces(image)
                self.predict_faces(image, faces)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                camera.clear()
                camera.draw_faces(faces)
                camera.draw_time(elapsed_ms)
                camera.draw_count(len(faces))
                camera.update()
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
        return


@click.command()
@click.option('--width', type=int, default=1280)
@click.option('--height', type=int, default=720)
@click.option('--hflip/--no-hflip', is_flag=True, default=DEFAULT_HFLIP)
@click.option('--vflip/--no-vflip', is_flag=True, default=DEFAULT_VFLIP)
@click.option('--tpu/--no-tpu', is_flag=True, default=False)
@click.option('--threshold', type=float, default=0.5)
@click.option('--fontsize', type=int, default=20)
def main(
    width: int,
    height: int,
    hflip: bool,
    vflip: bool,
    tpu: bool,
    threshold: float,
    fontsize: int
) -> None:
    face_model_path = 'models/mobilenet_ssd_v2_'
    if tpu:
        face_model_path += 'face_quant_postprocess_edgetpu.tflite'
    else:
        face_model_path += 'face_quant_postprocess.tflite'
    agender_model_path = 'agender/agender'
    if tpu:
        agender_model_path += '_edgetpu.tflite'
    else:
        agender_model_path += '.tflite'
    assert(os.path.exists(face_model_path))
    assert(os.path.exists(agender_model_path))
    width, height = _round_buffer_dims((width, height))
    detector = Detector(
        width=width,
        height=height,
        hflip=hflip,
        vflip=vflip,
        face_model_path=face_model_path,
        agender_model_path=agender_model_path,
        threshold=threshold,
        fontsize=fontsize
    )
    detector.run()
    return


if __name__ == "__main__":
    main()
