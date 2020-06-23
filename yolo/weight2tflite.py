#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ZeroPadding2D,
    Add,
    UpSampling2D,
    MaxPooling2D,
    Concatenate,
    Activation,
    Input,
    Lambda
)
from tensorflow.keras import backend as K
from absl import app, flags, logging
from absl.flags import FLAGS
import os
from collections import OrderedDict
import configparser
import numpy as np

NUM_CLASS = 80
IMAGE_SIZE = 320
NUM_TRAINING_IMAGES = 256
MODELS = ['yolov3-tiny', 'yolov3', 'yolov4']
MODES = ['dynamic', 'full']
flags.DEFINE_string('model', 'yolov3-tiny', 'model name')
flags.DEFINE_string('mode', 'dynamic', 'quantize mode')
flags.register_validator(
    'model',
    lambda x: x in MODELS,
    f'model should be in {MODELS}',
    flag_values=FLAGS
)
flags.register_validator(
    'mode',
    lambda x: x in MODES,
    f'mode should be in {MODES}',
    flag_values=FLAGS
)


def Mish(x):
    return Lambda(
        lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x)))
    )(x)


def darknet_conv(x, layer, rf):
    filters = int(layer['filters'])
    size = int(layer['size'])
    stride = int(layer['stride'])
    pad = int(layer['pad'])
    activation = layer['activation']
    batch_normalize = 'batch_normalize' in list(layer.keys())
    if pad == 1 and stride == 1:
        padding = 'SAME'
    else:
        padding = 'VALID'
    prev_layer_shape = K.int_shape(x)
    weights_shape = (size, size, prev_layer_shape[-1], filters)
    darknet_w_shape = (filters, prev_layer_shape[-1], size, size)
    weights_size = np.product(weights_shape)
    conv_bias = np.ndarray(
        shape=(filters,),
        dtype=np.float32,
        buffer=rf.read(filters * 4)
    )
    if batch_normalize:
        bn_weights = np.ndarray(
            shape=(3, filters),
            dtype=np.float32,
            buffer=rf.read(filters * 12)
        )
        bn_weight_list = [
            bn_weights[0],  # scale gamma
            conv_bias,      # shift beta
            bn_weights[1],  # running mean
            bn_weights[2]   # running var
        ]
    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype=np.float32,
        buffer=rf.read(weights_size * 4)
    )
    # DarkNet conv_weights are serialized Caffe-style:
    # (out_dim, in_dim, height, width)
    # We would like to set these to Tensorflow order:
    # (height, width, in_dim, out_dim)
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
    if batch_normalize:
        conv_weights = [conv_weights]
    else:
        conv_weights = [conv_weights, conv_bias]
    act_fn = None
    if activation in ['leaky', 'relu', 'mish', 'linear']:
        pass
    else:
        raise NotImplementedError(f'{activation} is not implemented')
    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=stride,
        padding=padding,
        use_bias=not batch_normalize,
        weights=conv_weights,
        activation=act_fn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0)
    )(x)
    if batch_normalize:
        x = BatchNormalization(
            weights=bn_weight_list
        )(x)
    if FLAGS.mode == 'dynamic':
        if activation == 'leaky':
            x = tf.nn.leaky_relu(x)
        elif activation == 'mish':
            x = Mish(x)
        elif activation == 'relu':
            x = Activation('relu')(x)
    else:
        if activation == 'leaky':
            x = Activation('relu')(x)
        elif activation == 'mish':
            x = Activation('relu')(x)
        elif activation == 'relu':
            x = Activation('relu')(x)
    return x


class LayerDict(OrderedDict):
    def __init__(self):
        super().__init__()
        self.seq = 0
        return

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            if key == 'net':
                return
            self.seq += 1
            key += ':%d' % self.seq
        super().__setitem__(key, val)
        return


def create_model(input_layer, config_fname, weight_fname):
    # read config file
    model_layers = configparser.ConfigParser(
        defaults=None, dict_type=LayerDict,
        strict=False, empty_lines_in_values=False
    )
    model_layers.read(config_fname)
    # read weights file
    rf = open(weight_fname, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3,), dtype=np.int32, buffer=rf.read(12)
    )
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        _ = np.ndarray(shape=(1,), dtype=np.int64, buffer=rf.read(8))
    else:
        _ = np.ndarray(shape=(1,), dtype=np.int32, buffer=rf.read(4))
    # create model
    all_layers = list()
    prev_layer = input_layer
    output_indexes = list()
    for section in model_layers.sections():
        layer = dict(model_layers.items(section=section))
        if section.startswith('convolutional'):
            all_layers.append(
                darknet_conv(prev_layer, layer, rf)
            )
            prev_layer = all_layers[-1]
        elif section.startswith('maxpool'):
            size = int(layer['size'])
            stride = int(layer['stride'])
            all_layers.append(MaxPooling2D(
                pool_size=(size, size),
                strides=(stride, stride),
                padding='SAME'
            )(prev_layer))
            prev_layer = all_layers[-1]
        elif section.startswith('upsample'):
            stride = int(layer['stride'])
            assert(stride == 2)
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]
        elif section.startswith('route'):
            ids = [int(i) for i in layer['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                concat_layer = Concatenate()(layers)
                all_layers.append(concat_layer)
                prev_layer = concat_layer
            else:
                skip_layer = layers[0]
                all_layers.append(skip_layer)
                prev_layer = skip_layer
        elif section.startswith('shortcut'):
            index = int(layer['from'])
            activation = layer['activation']
            assert(activation == 'linear')
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]
        elif section.startswith('yolo'):
            output_indexes.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = None
        else:
            raise NotImplementedError(f'section {section}')
    assert len(rf.read()) == 0, 'failed to read all data'
    rf.close()
    if len(output_indexes) == 0:
        output_indexes.append(len(all_layers) - 1)
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=[all_layers[i] for i in output_indexes]
    )
    return model


def representative_data_gen():
    for i in range(NUM_TRAINING_IMAGES):
        img = np.random.randint(
            0, NUM_TRAINING_IMAGES,
            (IMAGE_SIZE, IMAGE_SIZE, 3)
        )
        img = np.expand_dims(img, 0)
        img = np.array((img / 255.0), dtype=np.float32)
        if i % 10 == 9:
            logging.info("%d" % i)
        yield [img]


def save_tflite():
    weight = FLAGS.model + '.weights'
    assert(os.path.exists(weight))
    config = FLAGS.model + '.cfg'
    assert(os.path.exists(config))
    input_layer = Input(
        batch_size=1,
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    model = create_model(input_layer, config, weight)
    # model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if FLAGS.mode == 'full':
        output = FLAGS.model + '_full_quantized.tflite'
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        # converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
    else:
        output = FLAGS.model + '_dynamic_quantized.tflite'
    if tf.__version__ >= '2.2.0':
        converter.experimental_new_converter = False
    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)
    logging.info("tflite model is saved at {}".format(output))
    return output


def print_detail(details):
    for i, detail in enumerate(details):
        print("{}: index={} shape={} quantization={}, dtype={}".format(
            i, detail['index'], detail['shape'],
            detail['quantization'], detail['dtype']
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
