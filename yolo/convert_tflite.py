#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import tensorflow as tf
import os
import time
import numpy as np
import glob
import cv2
from yolo_models import YoloV3, YoloV3_tiny, YoloV4

NUM_CLASS = 80
IMAGE_SIZE = 320
NUM_TRAINING_IMAGES = 100
MODEL_CLASS = {
    'yolov3-tiny': YoloV3_tiny,
    'yolov3': YoloV3,
    'yolov4': YoloV4,
}
MODEL_SHAPE = {
    'yolov3-tiny': {
        'nlayers': 13,
        'nobn_layers': [9, 12],
    },
    'yolov3': {
        'nlayers': 75,
        'nobn_layers': [58, 66, 74],
    },
    'yolov4': {
        'nlayers': 110,
        'nobn_layers': [93, 101, 109],
    },
}


def load_darknet_weights(model: str, model_keras: tf.keras.Model) -> None:
    wf = open(f'{model}.weights', 'rb')
    major, minor, revision, seen, _ = np.fromfile(
        wf, dtype=np.int32, count=5
    )
    nlayers = MODEL_SHAPE[model]['nlayers']
    nobn_layers = MODEL_SHAPE[model]['nobn_layers']
    j = 0
    assert len(model_keras.weighted_layers) == nlayers
    for i, layers in enumerate(model_keras.weighted_layers):
        conv_layer = layers.conv
        norm_layer = layers.norm
        input_shape = layers.input_shape
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = input_shape[-1]
        if i not in nobn_layers:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(
                wf, dtype=np.float32, count=(4 * filters)
            )
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            j += 1
        else:
            conv_bias = np.fromfile(
                wf, dtype=np.float32, count=filters
            )
        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape)
        )
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose(
            [2, 3, 1, 0]
        )
        if i not in nobn_layers:
            assert norm_layer.__class__.__name__ == 'BatchNormalization'
            conv_layer.set_weights([conv_weights])
            norm_layer.set_weights(bn_weights)
        else:
            assert norm_layer.__class__.__name__ == 'function'
            conv_layer.set_weights([conv_weights, conv_bias])
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    return


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


def export_tflite_int8(model: str, model_keras: tf.keras.Model) -> None:
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
            scale = min(IMAGE_SIZE / ih, IMAGE_SIZE / iw)
            nh = int(ih * scale)
            nw = int(iw * scale)
            oh = (IMAGE_SIZE - nh) // 2
            ow = (IMAGE_SIZE - nw) // 2
            if scale >= 1:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            nimg = cv2.resize(
                img.copy(), (nw, nh),
                interpolation=interpolation
            )
            rimg = np.full(
                (IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8
            )
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
        # tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def export_tflite(model: str) -> None:
    if not os.path.isfile(f'{model}.weights'):
        print(f'ERROR: {model}.weights not found')
        return
    # load model
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  # NHWC
    input_shape = (1, *image_shape)
    model_keras = MODEL_CLASS[model](nc=NUM_CLASS)
    model_keras.build(input_shape=input_shape)
    # dummy run
    dummy_image_tf = np.random.randint(256, size=input_shape)
    dummy_image_tf = (dummy_image_tf / 255.0).astype(np.float32)
    dummy_image_tf = tf.convert_to_tensor(
        dummy_image_tf, dtype=np.float32
    )
    _ = model_keras(dummy_image_tf)
    # model_keras.summary()
    # load weights and export flat buffers
    load_darknet_weights(model=model, model_keras=model_keras)
    export_tflite_fp32(model=model, model_keras=model_keras)
    export_tflite_fp16(model=model, model_keras=model_keras)
    export_tflite_int8(model=model, model_keras=model_keras)
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


if __name__ == '__main__':
    # convert to tflite
    for model in ['yolov3-tiny', 'yolov3', 'yolov4']:
        export_tflite(model=model)
    # test tflite
    for model in ['yolov3-tiny', 'yolov3', 'yolov4']:
        for mode in ['fp32', 'fp16', 'int8']:
            test_tflite(model=model, mode=mode)
