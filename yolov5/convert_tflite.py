#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict
import os
import time
import argparse
import yaml
import glob
import cv2
import numpy as np
# Torch
import torch
# TensorFlow
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2
)
# YOLO V5
from yolov5_models import tf_Model, tf_Detect

NUM_TRAINING_IMAGES = 100
IMAGE_SIZE = 640


def convert_tf_keras_model(
    model: str,
    imgsize: List[int, int],
    model_torch: torch.nn.Module,
    nclasses: int,
    config: Dict
) -> tf.keras.Model:
    # convert PyTorch weights to TensorFlow
    model_tf = tf_Model(
        model_torch=model_torch,
        nclasses=nclasses,
        config=config
    )
    m = model_tf.model.layers[-1]
    assert isinstance(m, tf_Detect), "the last layer must be Detect"
    m.training = False
    # dummy run and check output
    dummy_image_tf = tf.zeros((1, *imgsize, 3))  # NHWC
    y = model_tf.predict(dummy_image_tf)
    for yy in y:
        _ = yy.numpy()
    # create keras model
    inputs_keras = tf.keras.Input(
        shape=(*imgsize, 3),
        batch_size=1
    )
    outputs_keras = model_tf.predict(inputs=inputs_keras)
    model_keras = tf.keras.Model(
        inputs=inputs_keras,
        outputs=outputs_keras,
        name=model
    )
    # model_keras.summary()
    return model_keras


def export_tflite_fp32(model: str, model_keras: tf.keras.Model) -> None:
    path_tflite = f'{model}_fp32.tflite'
    if os.path.isfile(path_tflite):
        return
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def export_tflite_fp16(model: str, model_keras: tf.keras.Model) -> None:
    path_tflite = f'{model}_fp16.tflite'
    if os.path.isfile(path_tflite):
        return
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    converter.target_spec.supported_types = [
        tf.float16
    ]
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def export_tflite_int8(
    model: str,
    imgsize: List[int, int],
    model_keras: tf.keras.Model
) -> None:
    path_tflite = f'{model}_int8.tflite'
    if os.path.isfile(path_tflite):
        return

    def representative_dataset_gen():
        images = glob.glob('val2017/*.jpg')
        np.random.shuffle(images)
        for i, ipath in enumerate(images[:NUM_TRAINING_IMAGES]):
            img = cv2.imread(ipath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ih = img.shape[0]
            iw = img.shape[1]
            scale = min(imgsize[0] / ih, imgsize[1] / iw)
            nh = int(ih * scale)
            nw = int(iw * scale)
            oh = (imgsize[0] - nh) // 2
            ow = (imgsize[1] - nw) // 2
            if scale >= 1:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            nimg = cv2.resize(
                img.copy(), (nw, nh),
                interpolation=interpolation
            )
            rimg = np.full((*imgsize, 3), 128, dtype=np.uint8)
            rimg[oh:oh + nh, ow:ow + nw, :] = nimg
            rimg = rimg[np.newaxis, ...].astype(np.float32)
            rimg /= 255.0
            yield [rimg]
            if i % 10 == 9:
                print(f'post-training... ({i}/{NUM_TRAINING_IMAGES})')
        return

    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def export_tflite(model: str, imgsize: List[int, int]) -> None:
    weights_torch = f'{model}.pt'
    if not os.path.isfile(weights_torch):
        print(f'ERROR: {weights_torch} not found')
        return
    # dummy image
    dummy_image_torch = torch.zeros((1, 3, *imgsize))  # NCHW
    # Load PyTorch model
    weights_torch = f'{model}.pt'
    model_torch = torch.load(
        weights_torch,
        map_location='cpu'
    )['model'].float()  # .fuse()
    model_torch.eval()
    # export=True to export Detect Layer
    model_torch.model[-1].export = False
    # dry run
    y = model_torch(dummy_image_torch)
    # number of classes
    nclasses = y[0].shape[-1] - 5
    # load configuration for the model
    path_config = f'models/{model}.yaml'
    with open(path_config, 'rt') as rf:
        config = yaml.safe_load(rf)
    # TensorFlow Keras export
    model_keras = convert_tf_keras_model(
        model=model,
        imgsize=imgsize,
        model_torch=model_torch,
        nclasses=nclasses,
        config=config
    )
    # TFLite model export
    export_tflite_fp32(model=model, model_keras=model_keras)
    export_tflite_fp16(model=model, model_keras=model_keras)
    export_tflite_int8(
        model=model, imgsize=imgsize, model_keras=model_keras
    )
    return


def print_detail(details: List) -> None:
    for i, detail in enumerate(details):
        print("{}: index={} shape={} dtype={}".format(
            i, detail['index'], detail['shape'], detail['dtype']
        ))
    return


def test_tflite(model: str, mode: str) -> None:
    tflite = f'{model}_{mode}.tflite'
    if not os.path.isfile(tflite):
        print(f'ERROR: {tflite} not found')
        return
    print(f'MODEL: {tflite}')
    interpreter = tf.lite.Interpreter(tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print('input details')
    print_detail(input_details)
    output_details = interpreter.get_output_details()
    print('output details')
    print_detail(output_details)
    input_shape = input_details[0]['shape']
    input_data = np.array(
        np.random.randint(0, 256, input_shape)
    )
    if mode == 'int8':
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data / 255.0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    elapsed = round((end_time - start_time) * 1000, 3)
    print(f'elapsed time taken for inference: {elapsed}[ms]')
    output_data = [
        interpreter.get_tensor(
            output_details[i]['index']
        ) for i in range(len(output_details))
    ]
    for i, out in enumerate(output_data):
        out_shape = output_details[i]['shape']
        assert len(out.shape) == len(out_shape)
        for j, v in enumerate(out.shape):
            assert v == out_shape[j]
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--imgsize', nargs='+', type=int,
        default=[IMAGE_SIZE, IMAGE_SIZE],
        help='image size'
    )  # height, width
    args = parser.parse_args()
    if len(args.imgsize) == 1:
        imgsize = args.imgsize * 2
    else:
        imgsize = args.imgsize[:2]
    for x in ['s', 'm', 'l', 'x']:
        export_tflite(model=f'yolov5{x}', imgsize=imgsize)
    for x in ['s', 'm', 'l', 'x']:
        for mode in ['fp32', 'fp16', 'int8']:
            test_tflite(model=f'yolov5{x}', mode=mode)
