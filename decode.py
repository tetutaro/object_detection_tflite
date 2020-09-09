#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional
import os
import re
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

EXTRACT_PAREN = re.compile(r'(?<=\().+?(?=\))')


# Intersection of Union
# boxesX is numpy array of
# offset 0: min x of bounding box
# offset 1: min y of bounding box
# offset 2: max x (x + width) of bounding box
# offset 3: max y (y + height) of bounding box
def bboxes_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    left_up = np.maximum(boxes1[:, :2], boxes2[:, :2])
    right_down = np.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = np.maximum(right_down - left_up, 0.0)
    inter_area = intersection[:, 0] * intersection[:, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(
        1.0 * inter_area / union_area,
        np.finfo(np.float32).eps
    )
    return ious


# Non-Maximum Supression
# https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
# bboxes is numpy array of
# offset 0: min x of bounding box
# offset 1: min y of bounding box
# offset 2: max x (x + width) of bounding box
# offset 3: max y (y + height) of bounding box
# offset 4: class ID (int)
# offset 5: probability
def non_maximum_supression(bboxes: np.array) -> List:
    unique_class_ids = list(set(bboxes[:, 4]))
    best_bboxes = list()
    for cls in unique_class_ids:
        cls_mask = (bboxes[:, 4] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 5])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]]
            )
            iou = bboxes_iou(
                np.array(best_bbox[np.newaxis, :4]),
                np.array(cls_bboxes[:, :4])
            )
            weight = np.ones((len(iou),), dtype=np.float32)
            iou_mask = iou > 0.213
            weight[iou_mask] = 0
            cls_bboxes[:, 5] = cls_bboxes[:, 5] * weight
            score_mask = cls_bboxes[:, 5] > 0
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


def sigmoid(x: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-x))


class Decoder(object):
    def __init__(
        self: Decoder,
        model_path: str,
        target: str,
        threshold: float,
        width: int,
        height: int
    ) -> None:
        # detect tpu
        if 'edgetpu' in model_path:
            self.is_tpu = True
        else:
            self.is_tpu = False
        # load labels
        if not os.path.exists('models/coco_labels.txt'):
            raise Exception("do download_models.sh")
        self.label2id = dict()
        self.id2label = dict()
        self.load_labels()
        # set target
        if target == 'all':
            self.target_id = -1
        else:
            if target not in list(self.label2id.keys()):
                raise ValueError(f'{target} not in labels')
            self.target_id = self.label2id[target]
        # load interpreter
        self.load_interpreter(model_path=model_path)
        # calculate size, scale, ...
        self.calc_scales(camera_width=width, camera_height=height)
        # others
        self.threshold = threshold
        return

    def load_labels(self: Decoder) -> None:
        pass

    def load_interpreter(self: Decoder, model_path: str) -> None:
        if self.is_tpu:
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
        return

    def calc_scales(
        self: Decoder,
        camera_width: int,
        camera_height: int
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
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

    def detect_objects(self: Decoder, image: Image) -> List:
        return []

    def get_bboxes(self: Decoder, outputs: List) -> List:
        return []


class DecoderSSD(Decoder):
    def __init__(
        self: DecoderSSD,
        model_path: str,
        target: str,
        threshold: float,
        width: int,
        height: int
    ) -> None:
        super().__init__(
            model_path=model_path, target=target, threshold=threshold,
            width=width, height=height
        )
        return

    def load_labels(self: DecoderSSD) -> None:
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

    def detect_objects(self: DecoderSSD, image: Image) -> List:
        # set input
        image = image.resize(self.c2i_size, Image.ANTIALIAS)
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
        return outputs

    def get_bboxes(self: DecoderSSD, outputs: List) -> List:
        assert(len(outputs) == 4)
        raw_boxes = outputs[0].squeeze()
        class_ids = outputs[1].squeeze()
        scores = outputs[2].squeeze()
        count = int(outputs[3].squeeze())
        bboxes = list()
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
            ymin, xmin, ymax, xmax = raw_boxes[i]
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
                bboxes.append({
                    'name': name,
                    'prob': prob,
                    'bbox': (xmin, ymin, xmax, ymax),
                })
        return bboxes


class DecoderYOLO(Decoder):
    def __init__(
        self: DecoderYOLO,
        model_path: str,
        target: str,
        threshold: float,
        width: int,
        height: int
    ) -> None:
        super().__init__(
            model_path=model_path, target=target, threshold=threshold,
            width=width, height=height
        )
        self.version = os.path.splitext(
            os.path.basename(model_path)
        )[0].split('_')[0]
        return

    def load_labels(self: DecoderYOLO) -> None:
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

    def detect_objects(self: DecoderYOLO, image: Image) -> List:
        # set input
        image = image.resize(self.c2i_size, Image.ANTIALIAS)
        background = Image.new(
            "RGB", (self.image_width, self.image_height), (128, 128, 128)
        )
        background.paste(image, self.c2i_offset)
        image = np.array(background, dtype=np.float32) / 255.0
        image = image[np.newaxis, ...]
        self.interpreter.set_tensor(
            self.input_index, image
        )
        # invoke
        self.interpreter.invoke()
        # get outputs
        outputs = [
            self.interpreter.get_tensor(i) for i in self.output_indexes
        ]
        return outputs

    def get_bboxes(
        self: DecoderYOLO,
        outputs: List
    ) -> List:
        if self.version == 'yolov3-tiny':
            stride_anchors = {
                16: [(10, 14), (23, 27), (37, 58)],
                32: [(81, 82), (135, 169), (344, 319)],
            }
            stride_xyscales = {16: 1.0, 32: 1.0}
        elif self.version == 'yolov3':
            stride_anchors = {
                8: [(10, 13), (16, 30), (33, 23)],
                16: [(30, 61), (62, 45), (59, 119)],
                32: [(116, 90), (156, 198), (373, 326)],
            }
            stride_xyscales = {8: 1.0, 16: 1.0, 32: 1.0}
        elif self.version == 'yolov4':
            stride_anchors = {
                8: [(12, 16), (19, 36), (40, 28)],
                16: [(36, 75), (76, 55), (72, 146)],
                32: [(142, 110), (192, 243), (459, 401)],
            }
            stride_xyscales = {8: 1.05, 16: 1.1, 32: 1.2}
        else:
            raise NotImplementedError()
        pred_bbox = list()
        for i, pred in enumerate(outputs):
            pred_shape = pred.shape
            pred_y = pred_shape[1]
            pred_x = pred_shape[2]
            strides = (self.image_width // pred_x, self.image_height // pred_y)
            stride = self.image_width // pred_x
            anchor = stride_anchors[stride]
            xyscale = stride_xyscales[stride]
            nc = len(self.id2label)
            pred = np.reshape(pred, (-1, pred_y, pred_x, 3, nc + 5))
            xy, wh, conf, prob = np.split(
                pred, (2, 4, 5), axis=-1
            )
            xy_offset = np.meshgrid(np.arange(pred_x), np.arange(pred_y))
            xy_offset = np.expand_dims(np.stack(xy_offset, axis=-1), axis=2)
            xy_offset = np.tile(
                np.expand_dims(xy_offset, axis=0), [1, 1, 1, 3, 1]
            ).astype(np.float)
            xy = ((
                sigmoid(xy) * xyscale
            ) - (0.5 * (xyscale - 1)) + xy_offset) * strides * self.i2c_scale
            wh = np.exp(wh) * anchor * self.i2c_scale
            conf = sigmoid(conf)
            prob = sigmoid(prob)
            pred_bbox.append(
                np.concatenate([xy, wh, conf, prob], axis=-1)
            )
        pred_bbox = [np.reshape(x, (-1, x.shape[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        # class_ids and scores
        conf = np.expand_dims(pred_bbox[:, 4], -1)
        prob = pred_bbox[:, 5:]
        prob = conf * prob
        if self.version == 'yolov3-tiny':
            prob = np.power(prob, 0.3)
        class_ids = np.argmax(prob, axis=-1)
        scores = np.max(prob, axis=-1)
        # xywh -> (xmin, ymin, xmax, ymax)
        xywh = pred_bbox[:, 0:4]
        boxes = np.concatenate([
            (xywh[:, :2] - (xywh[:, 2:] * 0.5)) - self.i2c_offset,
            (xywh[:, :2] + (xywh[:, 2:] * 0.5)) - self.i2c_offset
        ], axis=-1)
        # clip boxes those are out of range
        boxes = np.concatenate([
            np.maximum(boxes[:, :2], [0, 0]),
            np.minimum(boxes[:, 2:], [self.camera_width, self.camera_height])
        ], axis=-1)
        invalid_mask = np.logical_or(
            (boxes[:, 0] > boxes[:, 2]),
            (boxes[:, 1] > boxes[:, 3])
        )
        boxes[invalid_mask] = 0
        # discard invalid boxes
        box_scales = np.sqrt(np.multiply.reduce(
            boxes[:, 2:4] - boxes[:, 0:2], axis=-1
        ))
        scale_mask = np.logical_and(
            (0 < box_scales),
            (box_scales < np.inf)
        )
        score_mask = scores > self.threshold
        mask = np.logical_and(scale_mask, score_mask)
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]
        bboxes = np.concatenate([
            boxes, class_ids[:, np.newaxis], scores[:, np.newaxis]
        ], axis=-1)
        best_bboxes = non_maximum_supression(bboxes=bboxes)
        bboxes = list()
        for bbox in best_bboxes:
            cid = int(bbox[4])
            prob = float(bbox[5])
            name = self.id2label.get(cid)
            if name is None:
                continue
            if self.target_id != -1 and self.target_id != cid:
                continue
            if prob < self.threshold:
                continue
            xmin, ymin, xmax, ymax = bbox[0:4]
            xmin = max(0, int(xmin))
            xmax = min(self.camera_width, int(xmax))
            ymin = max(0, int(ymin))
            ymax = min(self.camera_height, int(ymax))
            bboxes.append({
                'name': name,
                'prob': prob,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        return bboxes


def get_decoder(
    model_path: str,
    target: str,
    threshold: float,
    width: int,
    height: int
) -> Decoder:
    if 'yolo' in model_path:
        return DecoderYOLO(
            model_path=model_path, target=target, threshold=threshold,
            width=width, height=height
        )
    else:
        return DecoderSSD(
            model_path=model_path, target=target, threshold=threshold,
            width=width, height=height
        )


class Predictor(Decoder):
    def __init__(
        self: Predictor,
        model_path: str
    ) -> None:
        # detect tpu
        if 'edgetpu' in model_path:
            self.is_tpu = True
        else:
            self.is_tpu = False
        # load interpreter
        self.load_interpreter(model_path=model_path)
        return

    def predict(self: Predictor, image: Image, objects: List) -> None:
        pass


class PredictorAgender(Predictor):
    def predict(self: PredictorAgender, image: Image, faces: List) -> None:
        for face in faces:
            xmin, ymin, xmax, ymax = face['bbox']
            faceimg = image.crop(
                (xmin, ymin, xmax, ymax)
            ).resize(
                (64, 64), Image.ANTIALIAS
            )
            faceimg = np.array(faceimg, dtype=np.float32)[np.newaxis, ...]
            self.interpreter.set_tensor(
                self.input_index, faceimg
            )
            self.interpreter.invoke()
            outputs = [
                self.interpreter.get_tensor(
                    i
                ) for i in self.output_indexes
            ]
            for output in outputs:
                output = np.squeeze(output)
                if output.shape[0] == 2:
                    # gender
                    if output[0] < 0.5:
                        gender = 'M'
                    else:
                        gender = 'F'
                else:
                    # age
                    age = output.dot(np.arange(0, 101).reshape(101, 1))
                    age = int(age[0])
            face['name'] = "%s-%d" % (gender, age)
        return


class PredictorMobileNet(Predictor):
    def __init__(
        self: PredictorMobileNet,
        model_path: str,
        target: Optional[str],
        threshold: Optional[float]
    ) -> None:
        super().__init__(model_path=model_path)
        self.id2label = dict()
        self.label2id = dict()
        self.load_labels(model_path=model_path)
        self.target = target
        if threshold is None:
            self.threshold = 0.5
        else:
            self.threshold = threshold
        return

    def load_labels(self: PredictorMobileNet, model_path: str) -> None:
        if 'bird' in model_path:
            label_path = 'models/inat_bird_labels.txt'
        elif 'insect' in model_path:
            label_path = 'models/inat_insect_labels.txt'
        elif 'plant' in model_path:
            label_path = 'models/inat_plant_labels.txt'
        else:
            label_path = 'models/imagenet_labels.txt'
        with open(label_path, 'rt') as rf:
            line = rf.readline()
            while line:
                info = line.strip().split(' ', 1)
                if len(info) != 2:
                    line = rf.readline()
                    continue
                idx = int(info[0])
                label = info[1].split(',')[0].strip().replace(' ', '_')
                match = re.findall(EXTRACT_PAREN, label)
                if len(match) > 0:
                    label = match[0]
                if label.endswith('_()'):
                    label = label[:-3]
                self.id2label[idx] = label
                self.label2id[label] = idx
                line = rf.readline()
        return

    def predict(self: PredictorMobileNet, image: Image, bboxes: List) -> List:
        return self._predict(image=image, bboxes=bboxes, imagesize=224)

    def _predict(
        self: PredictorMobileNet,
        image: Image,
        bboxes: List,
        imagesize: int
    ) -> List:
        objects = list()
        for bbox in bboxes:
            background = Image.new(
                "RGB", (imagesize, imagesize), (128, 128, 128)
            )
            x, y, w, h = bbox
            cropped = image.crop((x, y, x+w, y+h))
            if w >= h:
                adjust = int(imagesize * h / w)
                cropped = cropped.resize((imagesize, adjust), Image.BICUBIC)
                margin = (imagesize - adjust) // 2
                background.paste(cropped, (0, margin))
            else:
                adjust = int(imagesize * w / h)
                cropped = cropped.resize((adjust, imagesize), Image.BICUBIC)
                margin = (imagesize - adjust) // 2
                background.paste(cropped, (margin, 0))
            data = np.array(background, dtype=np.uint8)[np.newaxis, ...]
            self.interpreter.set_tensor(
                self.input_index, data
            )
            self.interpreter.invoke()
            outputs = [
                self.interpreter.get_tensor(
                    i
                ) for i in self.output_indexes
            ]
            assert(len(outputs) == 1)
            idx = np.argmax(outputs[0])
            label = self.id2label[idx]
            prob = outputs[0][0, idx] / np.sum(outputs)
            if label != 'background' and prob >= self.threshold:
                objects.append({
                    'name': label,
                    'index': idx,
                    'prob': prob,
                    'bbox': (x, y, x + w, y + h),
                })
        return objects


class PredictorInception(PredictorMobileNet):
    def predict(self: PredictorInception, image: Image, bboxes: List) -> List:
        return self._predict(image=image, bboxes=bboxes, imagesize=299)


def get_predictor(
    model_path: str,
    target: Optional[str] = None,
    threshold: Optional[float] = None
) -> Predictor:
    if 'agender' in model_path:
        return PredictorAgender(model_path=model_path)
    if 'inception' in model_path:
        return PredictorInception(
            model_path=model_path,
            target=target,
            threshold=threshold
        )
    else:
        return PredictorMobileNet(
            model_path=model_path,
            target=target,
            threshold=threshold
        )
