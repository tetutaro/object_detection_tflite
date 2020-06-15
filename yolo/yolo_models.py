#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    ZeroPadding2D,
    Lambda,
    MaxPool2D
)


def Mish(x) -> tf.Tensor:
    return Lambda(
        lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x)))
    )(x)


def DarknetConv(
    x: tf.Tensor,
    fil: int,
    ksize: int,
    act: bool = True,
    actfunc: str = 'leaky',
    ds: bool = False,
    bn: bool = True
) -> tf.Tensor:
    if ds:
        # downsampling
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        strides = 2
        padding = 'VALID'
    else:
        strides = 1
        padding = 'SAME'
    x = tf.keras.layers.Conv2D(
        filters=fil,
        kernel_size=ksize,
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
    )(x)
    if bn:
        # batch normalization
        x = BatchNormalization()(x)
    if act:
        # activation
        if actfunc == 'leaky':
            x = tf.nn.leaky_relu(x, alpha=0.1)
        elif actfunc == 'mish':
            x = Mish(x)
    return x


def UpSampling(x: tf.Tensor) -> tf.Tensor:
    double_shape = (x.shape[1] * 2, x.shape[2] * 2)
    return tf.image.resize(x, double_shape, method='nearest')


def DarknetResidual(
    x: tf.Tensor,
    fils: Tuple[int],
    actfunc: str = 'leaky'
) -> tf.Tensor:
    '''Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    '''
    f1, f2 = fils
    short_cut = x
    x = DarknetConv(x, fil=f2, ksize=1, actfunc=actfunc)
    x = DarknetConv(x, fil=f1, ksize=3, actfunc=actfunc)
    return short_cut + x


def DarknetBlock(
    x: tf.Tensor,
    fils: Tuple[int],
    blocks: int
) -> tf.Tensor:
    f1, f2 = fils
    x = DarknetConv(x, fil=f1, ksize=3, ds=True)
    for _ in range(blocks):
        x = DarknetResidual(x, fils=fils)
    return x


def Darknet53_tiny(x: tf.Tensor) -> Tuple[tf.Tensor]:
    x = DarknetConv(x, fil=16, ksize=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)
    x = DarknetConv(x, fil=32, ksize=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)
    x = DarknetConv(x, fil=64, ksize=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)
    x = DarknetConv(x, fil=128, ksize=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)
    x = DarknetConv(x, fil=256, ksize=3)
    x_8 = x
    x = MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)
    x = DarknetConv(x, fil=512, ksize=3)
    x = MaxPool2D(pool_size=2, strides=1, padding='SAME')(x)
    x = DarknetConv(x, fil=1024, ksize=3)
    return x_8, x


def Darknet53(x: tf.Tensor) -> Tuple[tf.Tensor]:
    '''YOLOv3
    https://arxiv.org/abs/1804.02767
    '''
    x = DarknetConv(x, fil=32, ksize=3)
    x = DarknetBlock(x, fils=(64, 32), blocks=1)
    x = DarknetBlock(x, fils=(128, 64), blocks=2)
    x = DarknetBlock(x, fils=(256, 128), blocks=8)
    x_36 = x
    x = DarknetBlock(x, fils=(512, 256), blocks=8)
    x_61 = x
    x = DarknetBlock(x, fils=(1024, 512), blocks=4)
    return x_36, x_61, x


def Darknet53_CSPnet(x):
    '''YOLOv4
    https://arxiv.org/abs/2004.10934
    '''
    x = DarknetConv(x, fil=32, ksize=3, actfunc='mish')
    # Darknet block with CSPnet: 1
    # Cross Stage Partial Network
    # https://arxiv.org/abs/1911.11929
    route_1 = DarknetConv(x, fil=64, ksize=3, ds=True, actfunc='mish')
    route_2 = route_1
    route_2 = DarknetConv(route_2, fil=64, ksize=1, actfunc='mish')
    route_1 = DarknetConv(route_1, fil=64, ksize=1, actfunc='mish')
    route_1 = DarknetResidual(route_1, fils=(64, 32), actfunc='mish')
    route_1 = DarknetConv(route_1, fil=64, ksize=1, actfunc='mish')
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(x, fil=64, ksize=1, actfunc='mish')
    # Darknet block with CSPnet: 2
    route_1 = DarknetConv(x, fil=128, ksize=3, ds=True, actfunc='mish')
    route_2 = route_1
    route_2 = DarknetConv(route_2, fil=64, ksize=1, actfunc='mish')
    route_1 = DarknetConv(route_1, fil=64, ksize=1, actfunc='mish')
    for _ in range(2):
        route_1 = DarknetResidual(route_1, fils=(64, 64), actfunc='mish')
    route_1 = DarknetConv(route_1, fil=64, ksize=1, actfunc='mish')
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(x, fil=128, ksize=1, actfunc='mish')
    # Darknet block with CSPnet: 3
    route_1 = DarknetConv(x, fil=256, ksize=3, ds=True, actfunc='mish')
    route_2 = route_1
    route_2 = DarknetConv(route_2, fil=128, ksize=1, actfunc='mish')
    route_1 = DarknetConv(route_1, fil=128, ksize=1, actfunc='mish')
    for _ in range(8):
        route_1 = DarknetResidual(route_1, fils=(128, 128), actfunc='mish')
    route_1 = DarknetConv(route_1, fil=128, ksize=1, actfunc='mish')
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(x, fil=256, ksize=1, actfunc='mish')
    # short cut
    x_54 = x
    # Darknet block with CSPnet: 4
    route_1 = DarknetConv(x, fil=512, ksize=3, ds=True, actfunc='mish')
    route_2 = route_1
    route_2 = DarknetConv(route_2, fil=256, ksize=1, actfunc='mish')
    route_1 = DarknetConv(route_1, fil=256, ksize=1, actfunc='mish')
    for _ in range(8):
        route_1 = DarknetResidual(route_1, fils=(256, 256), actfunc='mish')
    route_1 = DarknetConv(route_1, fil=256, ksize=1, actfunc='mish')
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(x, fil=512, ksize=1, actfunc='mish')
    # short cut
    x_85 = x
    # Darknet block with CSPnet: 5
    route_1 = DarknetConv(x, fil=1024, ksize=3, ds=True, actfunc='mish')
    route_2 = route_1
    route_2 = DarknetConv(route_2, fil=512, ksize=1, actfunc='mish')
    route_1 = DarknetConv(route_1, fil=512, ksize=1, actfunc='mish')
    for _ in range(4):
        route_1 = DarknetResidual(route_1, fils=(512, 512), actfunc='mish')
    route_1 = DarknetConv(route_1, fil=512, ksize=1, actfunc='mish')
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(x, fil=1024, ksize=1, actfunc='mish')
    # SPP block
    # Spatial Pyramid Pooling
    # https://arxiv.org/abs/1406.4729
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    x = tf.concat([
        tf.nn.max_pool2d(x, ksize=13, strides=1, padding='SAME'),
        tf.nn.max_pool2d(x, ksize=9, strides=1, padding='SAME'),
        tf.nn.max_pool2d(x, ksize=5, strides=1, padding='SAME'),
        x
    ], axis=-1)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    return x_54, x_85, x


def YoloLayer(x: tf.Tensor, nc: int) -> tf.Tensor:
    x = DarknetConv(x, fil=3 * (nc + 5), ksize=1, act=False, bn=False)
    return x


def YoloV3_tiny(input_layer: tf.Tensor, nc: int) -> tf.Tensor:
    x_8, x = Darknet53_tiny(input_layer)
    x = DarknetConv(x, fil=256, ksize=1)
    pred = x
    pred = DarknetConv(pred, fil=512, ksize=3)
    large_bbox = YoloLayer(pred, nc=nc)
    x = DarknetConv(x, fil=128, ksize=1)
    x = UpSampling(x)
    x = tf.concat([x, x_8], axis=-1)
    x = DarknetConv(x, fil=256, ksize=3)
    middle_bbox = YoloLayer(x, nc=nc)
    return [middle_bbox, large_bbox]


def YoloV3(input_layer: tf.Tensor, nc: int) -> tf.Tensor:
    x_36, x_61, x = Darknet53(input_layer)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    pred = x
    pred = DarknetConv(pred, fil=1024, ksize=3)
    large_bbox = YoloLayer(pred, nc=nc)
    x = DarknetConv(x, fil=256, ksize=1)
    x = UpSampling(x)
    x = tf.concat([x, x_61], axis=-1)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    pred = x
    pred = DarknetConv(pred, fil=512, ksize=3)
    middle_bbox = YoloLayer(pred, nc=nc)
    x = DarknetConv(x, fil=128, ksize=1)
    x = UpSampling(x)
    x = tf.concat([x, x_36], axis=-1)
    x = DarknetConv(x, fil=128, ksize=1)
    x = DarknetConv(x, fil=256, ksize=3)
    x = DarknetConv(x, fil=128, ksize=1)
    x = DarknetConv(x, fil=256, ksize=3)
    x = DarknetConv(x, fil=128, ksize=1)
    x = DarknetConv(x, fil=256, ksize=3)
    small_bbox = YoloLayer(x, nc=nc)
    return [small_bbox, middle_bbox, large_bbox]


def YoloV4(input_layer: tf.Tensor, nc: int) -> tf.Tensor:
    x_54, x_85, x = Darknet53_CSPnet(input_layer)
    short_cut = x
    x = DarknetConv(x, fil=256, ksize=1)
    x = UpSampling(x)
    x_85 = DarknetConv(x_85, fil=256, ksize=1)
    x = tf.concat([x_85, x], axis=-1)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    x_85 = x
    x = DarknetConv(x, fil=128, ksize=1)
    x = UpSampling(x)
    x_54 = DarknetConv(x_54, fil=128, ksize=1)
    x = tf.concat([x_54, x], axis=-1)
    x = DarknetConv(x, fil=128, ksize=1)
    x = DarknetConv(x, fil=256, ksize=3)
    x = DarknetConv(x, fil=128, ksize=1)
    x = DarknetConv(x, fil=256, ksize=3)
    x = DarknetConv(x, fil=128, ksize=1)
    pred = x
    pred = DarknetConv(pred, fil=256, ksize=3)
    small_bbox = YoloLayer(pred,  nc=nc)
    x = DarknetConv(x, fil=256, ksize=3, ds=True)
    x = tf.concat([x, x_85], axis=-1)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    x = DarknetConv(x, fil=512, ksize=3)
    x = DarknetConv(x, fil=256, ksize=1)
    pred = x
    pred = DarknetConv(pred, fil=512, ksize=3)
    middle_bbox = YoloLayer(pred, nc=nc)
    x = DarknetConv(x, fil=512, ksize=3, ds=True)
    x = tf.concat([x, short_cut], axis=-1)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    x = DarknetConv(x, fil=512, ksize=1)
    x = DarknetConv(x, fil=1024, ksize=3)
    large_bbox = YoloLayer(x, nc=nc)
    return [small_bbox, middle_bbox, large_bbox]


def DetectionLayer(
    x: tf.Tensor,
    nc: int,
    channels: int = 3
) -> List[tf.Tensor]:
    shape = tf.shape(x)
    output = tf.reshape(x, (*shape[:3], channels, nc + 5))
    xy, wh, conf, prob = tf.split(output, (2, 2, 1, nc), axis=-1)
    xy = tf.sigmoid(xy)
    wh = tf.exp(wh)
    conf = tf.sigmoid(conf)
    prob = tf.sigmoid(prob)
    return tf.concat([xy, wh, conf, prob], axis=-1)
