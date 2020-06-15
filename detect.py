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
        model_path: str,
        target: str,
        threshold: float,
        fontsize: int
    ) -> None:
        if 'yolo' in model_path:
            self.is_yolo = True
            self.yolo_version = os.path.splitext(
                os.path.basename(model_path)
            )[0]
        else:
            self.is_yolo = False
        if 'edgetpu' in model_path:
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
                if self.is_yolo:
                    self.id2label[off] = label
                    self.label2id[label] = off
                else:
                    self.id2label[idx] = label
                    self.label2id[label] = idx
                line = rf.readline()
                off += 1
        # set target
        if target == 'all':
            self.target_id = -1
        else:
            if target not in self.label2id.keys():
                raise ValueError('target not in models/coco_labels.txt')
            self.target_id = self.label2id[target]
        # load interpreter
        if self.is_tpu:
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
        self.output_indexes = [
            output_details[i]['index'] for i in range(len(output_details))
        ]
        if self.is_yolo:
            self.get_frames = self.get_frames_yolo
        else:
            self.get_frames = self.get_frames_ssd
        # set paramters
        self.camera_width = width
        self.camera_height = height
        self.hflip = hflip
        self.vflip = vflip
        self.threshold = threshold
        self.fontsize = fontsize
        return

    def get_frames_ssd(self: Detector, outputs: List) -> List:
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
            if self.target_id != -1 and self.target_id != cid:
                continue
            if prob < self.threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = max(0, int(ymin * self.camera_height))
            xmin = max(0, int(xmin * self.camera_width))
            ymax = min(self.camera_height, int(ymax * self.camera_height))
            xmax = min(self.camera_width, int(xmax * self.camera_width))
            frames.append({
                'name': name,
                'prob': prob,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        if len(frames) > 0:
            print(frames)
        return frames

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1. / (1. + np.exp(-x))

    def get_frames_yolo(self: Detector, outputs: List) -> List:
        frames = list()
        if self.yolo_version == 'yolov3-tiny':
            strides = [16, 32]
            anchors = [
                [(10, 14), (23, 27), (37, 58)],
                [(81, 82), (135, 169), (344, 319)],
            ]
            xyscale = [1.0, 1.0]
        elif self.yolo_version == 'yolov3':
            strides = [8, 16, 32]
            anchors = [
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)],
            ]
            xyscale = [1.0, 1.0, 1.0]
        elif self.yolo_version == 'yolov4':
            strides = [8, 16, 32]
            anchors = np.array([
                [(12, 16), (19, 36), (40, 28)],
                [(36, 75), (76, 55), (72, 146)],
                [(142, 110), (192, 243), (459, 401)],
            ])
            xyscale = [1.2, 1.1, 1.05]
        else:
            raise NotImplementedError()
        assert(len(outputs) == len(strides))
        pred_bbox = list()
        for i, pred in enumerate(outputs):
            try:
                pred_shape = pred.shape
                pred_x = pred_shape[1]
                pred_y = pred_shape[2]
                raw_xy = pred[:, :, :, :, 0:2]
                raw_wh = pred[:, :, :, :, 2:4]
                others = pred[:, :, :, :, 4:]
                xy_grid = np.meshgrid(np.arange(pred_x), np.arange(pred_y))
                xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)
                xy_grid = np.tile(
                    np.expand_dims(xy_grid, axis=0),
                    [1, 1, 1, 3, 1]
                )
                xy_grid = xy_grid.astype(np.float)
                pred_xy = ((
                    self.sigmoid(raw_xy) * xyscale[i]
                ) - 0.5 * (xyscale[i] - 1) + xy_grid) * strides[i]
                pred_wh = (np.exp(raw_wh) * anchors[i])
                pred_bbox.append(
                    np.concatenate([pred_xy, pred_wh, others], axis=-1)
                )
            except Exception:
                continue
        pred_bbox = [np.reshape(x, (-1, x.shape[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        valid_scale = [0, np.inf]
        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        # (x, y, w, h) -> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([
            pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
            pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
        ], axis=-1)
        # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        resize_ratio = min(
            self.image_width / self.camera_width,
            self.image_height / self.camera_height
        )
        dw = (self.image_width - resize_ratio * self.camera_width) / 2
        dh = (self.image_height - resize_ratio * self.camera_height) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
        # clip some boxes those are out of range
        pred_coor = np.concatenate([
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [1, 1])
        ], axis=-1)
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]),
            (pred_coor[:, 1] > pred_coor[:, 3])
        )
        pred_coor[invalid_mask] = 0
        # discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(
                pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1
        ))
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale),
            (bboxes_scale < valid_scale[1])
        )
        # discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.threshold
        mask = np.logical_and(scale_mask, score_mask)
        boxes = pred_coor[mask]
        class_ids = classes[mask]
        scores = scores[mask]
        assert(len(boxes.shape) == 2)
        assert(len(class_ids.shape) == 1)
        assert(len(scores.shape) == 1)
        assert(boxes.shape[1] == 4)
        assert(boxes.shape[0] == class_ids.shape[0])
        assert(boxes.shape[0] == scores.shape[0])
        print(class_ids)
        if boxes.shape[0] == 0:
            return frames
        # utils.nms(bbox, 0,213, method='nms')
        # utils.draw_bbox
        return self.get_frames_ssd([boxes, class_ids, scores])

    def detect_objects(self: Detector, image: Image) -> List:
        # set input
        image = image.resize(
            (self.image_width, self.image_height),
            Image.ANTIALIAS
        )
        if self.is_yolo:
            image = np.array(image, dtype=np.float32)[np.newaxis, ...]
        else:
            image = np.array(image, dtype=np.uint8)[np.newaxis, ...]
        self.interpreter.set_tensor(
            self.input_index, image
        )
        # invoke
        self.interpreter.invoke()
        # get outputs
        outputs = [
            self.interpreter.get_tensor(i) for i in self.output_indexes
        ]
        # get frames from outputs
        frames = self.get_frames(outputs)
        return frames

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
                start_time = time.perf_counter()
                objects = self.detect_objects(image)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
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
        width=width,
        height=height,
        hflip=hflip,
        vflip=vflip,
        model_path=model_path,
        target=target,
        threshold=threshold,
        fontsize=fontsize
    )
    detector.run()
    return


if __name__ == "__main__":
    main()
