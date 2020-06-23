#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from model import WideResNet

flags.DEFINE_string(
    'mode', 'full', 'quantize mode (dynamic, full)'
)
NUM_TRAINING_IMAGES = 256


def representative_data_gen():
    for i in range(NUM_TRAINING_IMAGES):
        img = np.random.randint(
            0, NUM_TRAINING_IMAGES,
            (64, 64, 3)
        )
        img = np.expand_dims(img, 0)
        img = np.array(img, dtype=np.float32)
        if i % 10 == 9:
            logging.info("%d" % i)
        yield [img]


def save_tflite():
    output = 'agender.tflite'
    input_layer = tf.keras.layers.Input(
        batch_size=1,
        shape=[64, 64, 3]
    )
    agender = WideResNet(input_layer)
    model = tf.keras.Model(input_layer, agender)
    model.load_weights('weights.28-3.73.hdf5')
    # model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if FLAGS.mode == 'full':
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
    if tf.__version__ >= '2.2.0':
        converter.experimental_new_converter = False
    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)
    logging.info("tflite model is saved at {}".format(output))
    return output


def print_detail(details):
    for i, detail in enumerate(details):
        print("{}: index={} shape={} dtype={}".format(
            i, detail['index'], detail['shape'], detail['dtype']
        ))
    return


def demo(output):
    interpreter = tf.lite.Interpreter(output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')
    input_details = interpreter.get_input_details()
    print_detail(input_details)
    output_details = interpreter.get_output_details()
    print_detail(output_details)
    input_shape = input_details[0]['shape']
    input_data = np.array(
        np.random.randint(0, 256, input_shape), dtype=np.float32
    )
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [
        interpreter.get_tensor(
            output_details[i]['index']
        ) for i in range(len(output_details))
    ]
    for i, o in enumerate(output_data):
        print("{}th output: shape={}".format(i, o.shape))
    return


def main(_argv):
    output = save_tflite()
    demo(output)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
