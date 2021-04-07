#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict
import re
import numpy as np
from PIL import Image
from src.config import Config
from src.detection_model import DetectionModel


class PredictionModel(DetectionModel):
    def __init__(
        self: PredictionModel,
        config: Config,
        model_path: str
    ) -> None:
        self.config = config
        # load interpreter
        self.load_interpreter(model_path=model_path)
        return

    def predict(
        self: PredictionModel,
        image: Image,
        objects: List
    ) -> None:
        pass


class Agender(PredictionModel):
    def __init__(self: Agender, config: Config) -> None:
        if config.quant == 'tpu':
            quant = '_edgetpu'
        else:
            quant = ''
        model_path = f'agender/agender{quant}.tflite'
        super().__init__(config=config, model_path=model_path)
        return

    def predict(self: Agender, image: Image, objects: List) -> None:
        for face in objects:
            xmin, ymin, xmax, ymax = face['bbox']
            width = xmax - xmin
            height = ymax - ymin
            scale = min(64 / width, 64 / height)
            if scale > 1.0:
                fil = Image.BICUBIC
            elif scale < 1.0:
                fil = Image.LANCZOS
            faceimg = image.crop((xmin, ymin, xmax, ymax))
            new_size = (int(width * scale), int(height * scale))
            offset = ((64 - new_size[0]) // 2, (64 - new_size[1]) // 2)
            if scale != 1.0:
                faceimg = faceimg.resize(new_size, fil)
            background = Image.new(
                "RGB", (64, 64), (128, 128, 128)
            )
            background.paste(faceimg, offset)
            background = np.array(
                background, dtype=np.float32
            )[np.newaxis, ...]
            self.interpreter.set_tensor(
                self.input_index, background
            )
            self.interpreter.invoke()
            outputs = [
                self.interpreter.get_tensor(i)
                for i in self.output_indexes
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


class MobileNet(PredictionModel):
    def __init__(self: MobileNet, config: Config) -> None:
        if config.quant == 'tpu':
            quant = '_edgetpu'
        else:
            quant = ''
        if config.model == 'mobilenet':
            model = ''
        else:
            model = f'_inat_{config.model}'
        model_path = f'models/mobilenet_v2_1.0_224{model}'
        model_path += f'_quant{quant}.tflite'
        super().__init__(config=config, model_path=model_path)
        self.id2label = dict()
        self.label2id = dict()
        self.load_labels()
        if self.config.target == 'all':
            self.target_id = -1
        else:
            if self.config.target not in list(self.label2id.keys()):
                raise ValueError(f'{self.config.taget} not in labels')
            self.target_id = self.label2id[self.config.target]
        return

    def load_labels(self: MobileNet) -> None:
        EXTRACT_PAREN = re.compile(r'(?<=\().+?(?=\))')
        if self.config.model == 'bird':
            label_path = 'models/inat_bird_labels.txt'
        elif self.config.model == 'insect':
            label_path = 'models/inat_insect_labels.txt'
        elif self.config.model == 'plant':
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

    def predict(
        self: MobileNet,
        image: Image,
        objects: List,
        imagesize: Optional[int] = 224
    ) -> List[Dict]:
        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bbox']
            width = xmax - xmin
            height = ymax - ymin
            scale = min(imagesize / width, imagesize / height)
            if scale > 1.0:
                fil = Image.BICUBIC
            elif scale < 1.0:
                fil = Image.LANCZOS
            objimg = image.crop((xmin, ymin, xmax, ymax))
            new_size = (int(width * scale), int(height * scale))
            offset = (
                (imagesize - new_size[0]) // 2,
                (imagesize - new_size[1]) // 2
            )
            if scale != 1.0:
                objimg = objimg.resize(new_size, fil)
            background = Image.new(
                "RGB", (imagesize, imagesize), (128, 128, 128)
            )
            background.paste(objimg, offset)
            background = np.array(
                background, dtype=np.uint8
            )[np.newaxis, ...]
            self.interpreter.set_tensor(
                self.input_index, background
            )
            self.interpreter.invoke()
            outputs = [
                self.interpreter.get_tensor(i)
                for i in self.output_indexes
            ]
            assert(len(outputs) == 1)
            idx = np.argmax(outputs[0])
            label = self.id2label[idx]
            prob = outputs[0][0, idx] / np.sum(outputs)
            if (
                    label != 'background'
            ) and (
                prob >= self.config.prob_threshold
            ):
                obj['name'] = label
                obj['prob'] = prob
                obj['category_id'] = idx
        objects = [x for x in objects if x.get('name') is not None]
        return objects


def get_prediction_model(config: Config) -> PredictionModel:
    if config.model == 'agender':
        return Agender(config=config)
    else:
        return MobileNet(config=config)
