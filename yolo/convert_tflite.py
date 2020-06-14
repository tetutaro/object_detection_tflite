#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import glob
from yolo_models import YoloV3, YoloV3_tiny, YoloV4
from load_weights import load_weights_v3, load_weights_v3_tiny, load_weights_v4

NUM_CLASS = 80
IMAGE_SIZE = 416
MODEL_CLASS = {
    'yolov3': YoloV3,
    'yolov3-tiny': YoloV3_tiny,
    'yolov4': YoloV4,
}
WEIGHT_FUNC = {
    'yolov3': load_weights_v3,
    'yolov3-tiny': load_weights_v3_tiny,
    'yolov4': load_weights_v4,
}
flags.DEFINE_string(
    'weights', 'yolov3.weights', 'path to weights file'
)
flags.DEFINE_string(
    'mode', 'full', 'quantize mode (dynamic, full)'
)
NUM_TRAINING_IMAGES = 100


def representative_data_gen():
    files = glob.glob('coco/val2017/*.jpg')
    for fname in files[:NUM_TRAINING_IMAGES]:
        img = tf.image.decode_image(
            open(fname, 'rb').read(), channels=3
        )
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.expand_dims(img, 0)
        img = np.array(img, dtype=np.float32)
        logging.info(fname)
        yield [img]


def save_tflite():
    modelname = FLAGS.weights.split('.')[0]
    output = modelname + '.tflite'
    input_layer = tf.keras.layers.Input(
        batch_size=1,
        shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
        dtype=tf.float32
    )
    bbox_tensors = MODEL_CLASS[modelname](input_layer, NUM_CLASS)
    model = tf.keras.Model(input_layer, bbox_tensors)
    WEIGHT_FUNC[modelname](model, FLAGS.weights)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if FLAGS.mode == 'full':
        converter.target_spec.supported_ops = [
            # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # converter.inference_input_type = tf.uint8
        # converter.inference_type = tf.uint8
        converter.allow_custom_ops = True
        # converter.quantized_input_stats = {
        #     'inputs:0': (0., 255.)
        # }
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
