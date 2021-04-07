#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple
import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import platform
if platform.system() == 'Linux':  # RaspberryPi
    EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
elif platform.system() == 'Darwin':  # MacOS X
    EDGETPU_SHARED_LIB = 'libedgetpu.1.dylib'
else:
    raise NotImplementedError()
from src.config import Config
from src.utils import sigmoid, filter_bboxes

STRIDE_ANCHORS = {
    'yolov3-tiny': {
        16: [(10, 14), (23, 27), (37, 58)],
        32: [(81, 82), (135, 169), (344, 319)],
    },
    'yolov3': {
        8: [(10, 13), (16, 30), (33, 23)],
        16: [(30, 61), (62, 45), (59, 119)],
        32: [(116, 90), (156, 198), (373, 326)],
    },
    'yolov3-spp': {
        8: [(10, 13), (16, 30), (33, 23)],
        16: [(30, 61), (62, 45), (59, 119)],
        32: [(116, 90), (156, 198), (373, 326)],
    },
    'yolov4-tiny': {
        16: [(10, 14), (23, 27), (37, 58)],
        32: [(81, 82), (135, 169), (344, 319)],
    },
    'yolov4': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
    'yolov4-csp': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
    'yolov4x-mish': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
}
STRIDE_XYSCALES = {
    'yolov3-tiny': {16: 1.0, 32: 1.0},
    'yolov3': {8: 1.0, 16: 1.0, 32: 1.0},
    'yolov3-spp': {8: 1.0, 16: 1.0, 32: 1.0},
    'yolov4-tiny': {16: 1.05, 32: 1.05},
    'yolov4': {8: 1.05, 16: 1.1, 32: 1.2},
    'yolov4-csp': {8: 2.0, 16: 2.0, 32: 2.0},
    'yolov4x-mish': {8: 2.0, 16: 2.0, 32: 2.0},
}


class DetectionModel(object):
    def __init__(
        self: DetectionModel,
        config: Config,
        model_path: str
    ) -> None:
        self.config = config
        # load labels
        if not os.path.exists('models/coco_labels.txt'):
            raise Exception("do download_models.sh")
        self.label2id = dict()
        self.id2label = dict()
        self.load_labels()
        # set target
        if self.config.target == 'all':
            self.target_id = -1
        else:
            if self.config.target not in list(self.label2id.keys()):
                raise ValueError(f'{self.config.target} not in labels')
            self.target_id = self.label2id[self.config.target]
        # load interpreter
        self.load_interpreter(model_path=model_path)
        # calculate size, scale, ...
        self.calc_scales()
        return

    def load_labels(self: DetectionModel) -> None:
        pass

    def load_interpreter(self: DetectionModel, model_path: str) -> None:
        if self.config.quant == 'tpu':
            delegates = [
                tflite.load_delegate(EDGETPU_SHARED_LIB, {})
            ]
        else:
            delegates = None
        # load interpreter
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()
        # get input details
        input_detail = self.interpreter.get_input_details()[0]
        self.input_index = input_detail['index']
        shape = input_detail['shape']
        # set image size
        self.image_height = shape[1]
        self.image_width = shape[2]
        # get output details
        output_details = self.interpreter.get_output_details()
        self.output_indexes = [
            output_details[i]['index'] for i in range(len(output_details))
        ]
        self.output_quant_params = [
            output_details[i]['quantization_parameters']
            for i in range(len(output_details))
        ]
        return

    def calc_scales(self: DetectionModel) -> None:
        # camera2image
        self.c2i_scale = min(
            self.image_width / self.config.camera_width,
            self.image_height / self.config.camera_height
        )
        self.c2i_size = (
            int(self.config.camera_width * self.c2i_scale),
            int(self.config.camera_height * self.c2i_scale)
        )
        self.c2i_offset = (
            (self.image_width - self.c2i_size[0]) // 2,
            (self.image_height - self.c2i_size[1]) // 2
        )
        # image2camera
        self.i2c_scale = max(
            self.config.camera_width / self.image_width,
            self.config.camera_height / self.image_height
        )
        self.i2c_dummysize = (
            int(self.image_width * self.i2c_scale),
            int(self.image_height * self.i2c_scale)
        )
        self.i2c_offset = (
            (self.i2c_dummysize[0] - self.config.camera_width) // 2,
            (self.i2c_dummysize[1] - self.config.camera_height) // 2
        )
        return

    def inference(self: DetectionModel, image: Image) -> List[Dict]:
        return []


class SSD(DetectionModel):
    def __init__(self: SSD, config: Config) -> None:
        if config.model == 'ssd':
            data = 'coco'
        else:
            data = 'face'
        if config.quant == 'tpu':
            quant = '_edgetpu'
        else:
            quant = ''
        model_path = f'models/mobilenet_ssd_v2_{data}_quant'
        model_path += f'_postprocess{quant}.tflite'
        super().__init__(config=config, model_path=model_path)
        return

    def load_labels(self: SSD) -> None:
        with open('models/coco_labels.txt', 'rt', encoding='utf-8') as rf:
            line = rf.readline()
            while line:
                info = [x.strip() for x in line.strip().split(' ', 1)]
                idx = int(info[0])
                label = info[1].replace(' ', '_')
                self.id2label[idx] = label
                self.label2id[label] = idx
                line = rf.readline()
        return

    def inference(self: SSD, image: Image) -> List[Dict]:
        # set input
        if self.c2i_scale > 1.0:
            fil = Image.BICUBIC
        elif self.c2i_scale < 1.0:
            fil = Image.LANCZOS
        if self.c2i_scale != 1.0:
            image = image.resize(self.c2i_size, fil)
        background = Image.new(
            "RGB", (self.image_width, self.image_height), (128, 128, 128)
        )
        background.paste(image, self.c2i_offset)
        image = np.array(background, dtype=np.uint8)[np.newaxis, ...]
        self.interpreter.set_tensor(
            self.input_index, image
        )
        # invoke
        self.interpreter.invoke()
        # get outputs
        outputs = [
            self.interpreter.get_tensor(i) for i in self.output_indexes
        ]
        assert(len(outputs) == 4)
        # bounding boxes
        # 0-3: yxyx
        # 4: category id
        # 5: confidence score
        pred = np.squeeze(np.concatenate((
            outputs[0],                   # yxyx
            outputs[1][..., np.newaxis],  # category id
            outputs[2][..., np.newaxis]   # confidence score
        ), axis=-1), 0).copy()
        # number of valid bounding boxes
        count = int(outputs[3][0])
        if count < 20:
            pred = pred[:count, :]
        keep = pred[:, 5] >= self.config.conf_threshold
        if keep.sum() == 0:
            return []
        pred = pred[keep]
        if self.target_id != -1:
            keep = pred[:, 4] == self.target_id
            if keep.sum() == 0:
                return []
            pred = pred[keep]
        # adjust to original image size
        pred[:, 0] = np.maximum(
            (pred[:, 0] * self.i2c_dummysize[1]) - self.i2c_offset[1],
            0
        )
        pred[:, 1] = np.maximum(
            (pred[:, 1] * self.i2c_dummysize[0]) - self.i2c_offset[0],
            0
        )
        pred[:, 2] = np.minimum(
            (pred[:, 2] * self.i2c_dummysize[1]) - self.i2c_offset[1],
            self.config.camera_height
        )
        pred[:, 3] = np.minimum(
            (pred[:, 3] * self.i2c_dummysize[0]) - self.i2c_offset[0],
            self.config.camera_width
        )
        objects = list()
        for info in pred.tolist():
            ymin = int(info[0])
            xmin = int(info[1])
            ymax = int(info[2])
            xmax = int(info[3])
            cid = int(info[4])
            prob = float(info[5])
            name = self.id2label.get(cid)
            if name is None:
                continue
            objects.append({
                'name': name,
                'prob': prob,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        return objects


class YOLO(DetectionModel):
    def __init__(self: YOLO, config: Config) -> None:
        if config.quant == 'tpu':
            quant = 'int8_edgetpu'
        else:
            quant = config.quant
        model_path = f'models/{config.model}_{quant}.tflite'
        super().__init__(config=config, model_path=model_path)
        return

    def load_labels(self: YOLO) -> None:
        with open('models/coco_labels.txt', 'rt', encoding='utf-8') as rf:
            off = 0
            line = rf.readline()
            while line:
                info = [x.strip() for x in line.strip().split(' ', 1)]
                label = info[1].replace(' ', '_')
                self.id2label[off] = label
                self.label2id[label] = off
                line = rf.readline()
                off += 1
        return

    @staticmethod
    def apply_anchors_ver1(
        pred: np.ndarray,
        stride: int,
        anchor: List[Tuple[int]],
        xyscale: float
    ) -> np.ndarray:
        assert len(pred.shape) == 3
        assert pred.shape[2] == 255
        # align the axes to (x, y, anchor, data)
        pred = pred.reshape((*pred.shape[:2], 3, 85))
        # calc grid of anchor box
        anchor_grid = np.array(anchor)[np.newaxis, np.newaxis, :, :]
        grid = np.meshgrid(
            np.arange(pred.shape[1]), np.arange(pred.shape[0])
        )
        grid = np.stack(grid, axis=-1)[:, :, np.newaxis, :]
        # xy: min_x, min_y
        # wh: width, height
        # conf: confidence score of the bounding box
        # prob: probability for each category
        xy, wh, conf, prob = np.split(
            pred, (2, 4, 5), axis=-1
        )
        # apply anchor
        xy = (
            (sigmoid(xy) * xyscale) - (0.5 * (xyscale - 1)) + grid
        ) * stride
        wh = np.exp(wh) * anchor_grid
        conf = sigmoid(conf)
        prob = sigmoid(prob)
        # concat
        bbox = np.concatenate([xy, wh, conf, prob], axis=-1)
        # expand all anchors
        bbox = np.reshape(bbox, (-1, 85))
        return bbox

    @staticmethod
    def apply_anchors_ver2(
        pred: np.ndarray,
        stride: int,
        anchor: List[Tuple[int]],
        xyscale: float
    ) -> np.ndarray:
        assert len(pred.shape) == 3
        assert pred.shape[2] == 255
        # align the axes to (x, y, anchor, data)
        pred = pred.reshape((*pred.shape[:2], 3, 85))
        # calc grid of anchor box
        anchor_grid = np.array(anchor)[np.newaxis, np.newaxis, :, :]
        grid = np.meshgrid(
            np.arange(pred.shape[1]), np.arange(pred.shape[0])
        )
        grid = np.stack(grid, axis=-1)[:, :, np.newaxis, :]
        # apply anchor
        pred = sigmoid(pred)
        pred[..., :2] = (
            (pred[..., :2] * xyscale) - (0.5 * (xyscale - 1)) + grid
        ) * stride
        pred[..., 2:4] = ((pred[..., 2:4] * xyscale) ** 2) * anchor_grid
        # expand all anchors
        bbox = np.reshape(pred, (-1, 85))
        return bbox

    def apply_anchors(self: YOLO, preds: List[np.ndarray]) -> np.ndarray:
        anchors = STRIDE_ANCHORS[self.config.model]
        xyscales = STRIDE_XYSCALES[self.config.model]
        applied = list()
        for pred in preds:
            stride = max(
                self.image_width // pred.shape[0],
                self.image_height // pred.shape[1]
            )
            anchor = anchors[stride]
            xyscale = xyscales[stride]
            # call each version of apply_anchors
            if self.config.model in ['yolov4-csp', 'yolov4x-mish']:
                bbox = self.apply_anchors_ver2(
                    pred=pred,
                    stride=stride,
                    anchor=anchor,
                    xyscale=xyscale
                )
            else:
                bbox = self.apply_anchors_ver1(
                    pred=pred,
                    stride=stride,
                    anchor=anchor,
                    xyscale=xyscale
                )
            applied.append(bbox)
        return np.concatenate(applied, axis=0)

    def inference(self: YOLO, image: Image) -> List[Dict]:
        # set input
        if self.c2i_scale > 1.0:
            fil = Image.BICUBIC
        elif self.c2i_scale < 1.0:
            fil = Image.LANCZOS
        if self.c2i_scale != 1.0:
            image = image.resize(self.c2i_size, fil)
        background = Image.new(
            "RGB", (self.image_width, self.image_height), (128, 128, 128)
        )
        background.paste(image, self.c2i_offset)
        if self.config.is_int8:
            image = np.array(background, dtype=np.uint8)
        else:
            image = (
                np.array(background, dtype=np.float32) / 255.0
            ).astype(np.float32)
        image = image[np.newaxis, ...]
        self.interpreter.set_tensor(
            self.input_index, image
        )
        # invoke
        self.interpreter.invoke()
        # get outputs
        if self.config.is_int8:
            outputs = list()
            for index, params in zip(
                self.output_indexes, self.output_quant_params
            ):
                raw = self.interpreter.get_tensor(index)
                output = (
                    raw.astype(np.float32) - params['zero_points']
                ) * params['scales']
                outputs.append(output)
        else:
            outputs = [
                self.interpreter.get_tensor(i)
                for i in self.output_indexes
            ]  # List of np.array
        preds = [np.squeeze(x, 0).copy() for x in outputs]
        if self.config.model.startswith(('yolov3', 'yolov4')):
            pred = self.apply_anchors(preds=preds)
        else:
            pred = preds[0]
            pred[:, :4] = pred[:, :4] * max(
                self.image_height,
                self.image_width
            )
        assert len(pred.shape) == 2
        assert pred.shape[1] == 85
        # xywh -> xyxy
        xywh = pred[:, :4]
        xyxy = np.concatenate([
            (xywh[:, :2] - (xywh[:, 2:] * 0.5)),
            (xywh[:, :2] + (xywh[:, 2:] * 0.5))
        ], axis=-1)
        # rescale bouding boxes according to image preprocessing
        xyxy[:, 0] = np.maximum(
            (xyxy[:, 0] - self.c2i_offset[0]) * self.i2c_scale,
            0
        )
        xyxy[:, 1] = np.maximum(
            (xyxy[:, 1] - self.c2i_offset[1]) * self.i2c_scale,
            0
        )
        xyxy[:, 2] = np.minimum(
            (xyxy[:, 2] - self.c2i_offset[0]) * self.i2c_scale,
            self.config.camera_width
        )
        xyxy[:, 3] = np.minimum(
            (xyxy[:, 3] - self.c2i_offset[1]) * self.i2c_scale,
            self.config.camera_height
        )
        # confidence score of bbox and probability for each category
        conf = pred[:, 4:5]
        prob = pred[:, 5:]
        # confidence score for each category = conf * prob
        cat_conf = conf * prob
        if self.config.model == 'yolov3-tiny':
            cat_conf = np.power(cat_conf, 0.3)
        # catgory of bouding box is the most plausible category
        cat = cat_conf.argmax(axis=1)[:, np.newaxis].astype(np.float)
        # confidence score of bbox is that of the most plausible category
        conf = cat_conf.max(axis=1)[:, np.newaxis]
        # ready for NMS (0-3: xyxy, 4: category id, 5: confidence score)
        bboxes = np.concatenate((xyxy, cat, conf), axis=1)
        # filtering bounding boxes with NMS
        bboxes = filter_bboxes(
            bboxes=bboxes,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold
        )
        objects = list()
        for info in bboxes.tolist():
            xmin = int(info[0])
            ymin = int(info[1])
            xmax = int(info[2])
            ymax = int(info[3])
            cid = int(info[4])
            prob = float(info[5])
            name = self.id2label.get(cid)
            if name is None:
                continue
            objects.append({
                'name': name,
                'prob': prob,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        return objects


def get_detection_model(config: Config) -> DetectionModel:
    if config.model.startswith('yolo'):
        return YOLO(config=config)
    else:
        return SSD(config=config)
