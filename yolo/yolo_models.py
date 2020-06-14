#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    ZeroPadding2D,
    Lambda,
    LeakyReLU,
    MaxPool2D
)


def Mish(x) -> tf.Tensor:
    return Lambda(
        lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x)))
    )(x)


def DarknetConv(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    activation: bool = True,
    activation_type: str = 'leaky',
    downsampling: bool = False,
    batch_norm: bool = True
) -> tf.Tensor:
    if downsampling:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        strides = 2
        padding = 'valid'
    else:
        strides = 1
        padding = 'same'
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
    )(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if activation:
        if activation_type == 'leaky':
            x = LeakyReLU(alpha=0.1)(x)
        elif activation_type == 'mish':
            x = Mish(x)
    return x


def UpSampling(x: tf.Tensor) -> tf.Tensor:
    double_shape = (x.shape[1] * 2, x.shape[2] * 2)
    return tf.image.resize(x, double_shape, method='nearest')


def SPP_Block(x: tf.Tensor, filters: int) -> tf.Tensor:
    '''Spatial Pyramid Pooling: Extract feature map and resize it to fixed length
    https://arxiv.org/abs/1406.4729
    '''
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    x = DarknetConv(x, filters=filters, kernel_size=3)
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    x = tf.concat([
        MaxPool2D(pool_size=13, strides=1, padding='same')(x),
        MaxPool2D(pool_size=9, strides=1, padding='same')(x),
        MaxPool2D(pool_size=5, strides=1, padding='same')(x),
        x
    ], axis=-1)
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    x = DarknetConv(x, filters=filters, kernel_size=3)
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    return x


def DarknetResidual(
    x: tf.Tensor,
    filters_1: int,
    filters_2: int,
    activation_type: str = 'leaky'
) -> tf.Tensor:
    '''Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    '''
    short_cut = x
    x = DarknetConv(
        x, filters=filters_2, kernel_size=1,
        activation_type=activation_type
    )
    x = DarknetConv(
        x, filters=filters_1, kernel_size=3,
        activation_type=activation_type
    )
    return short_cut + x


def DarknetBlock(
    x: tf.Tensor,
    filters_1: int,
    filters_2: int,
    blocks: int
) -> tf.Tensor:
    x = DarknetConv(x, filters=filters_1, kernel_size=3, downsampling=True)
    for _ in range(blocks):
        x = DarknetResidual(x, filters_1=filters_1, filters_2=filters_2)
    return x


def DarknetBlock_CSPnet(
    x: tf.Tensor,
    filters_1: int,
    filters_2: int,
    blocks: int
) -> tf.Tensor:
    '''Cross Stage Partial Network
    https://arxiv.org/abs/1911.11929
    '''
    route_1 = DarknetConv(
        x, filters=filters_1, kernel_size=3, downsampling=True,
        activation_type='mish'
    )
    route_2 = route_1
    route_2 = DarknetConv(
        route_2, filters=filters_2, kernel_size=1,
        activation_type='mish'
    )
    route_1 = DarknetConv(
        route_1, filters=filters_2, kernel_size=1,
        activation_type='mish'
    )
    for _ in range(blocks):
        route_1 = DarknetResidual(
            route_1, filters_1=filters_2, filters_2=filters_2,
            activation_type='mish'
        )
    route_1 = DarknetConv(
        route_1, filters=filters_2, kernel_size=1,
        activation_type='mish'
    )
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(
        x, filters=filters_1, kernel_size=1,
        activation_type='mish'
    )
    return x


def Darknet53_tiny(x: tf.Tensor) -> Tuple[tf.Tensor]:
    x = DarknetConv(x, filters=16, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = DarknetConv(x, filters=32, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = DarknetConv(x, filters=64, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = DarknetConv(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = DarknetConv(x, filters=256, kernel_size=3)
    x_8 = x
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = DarknetConv(x, filters=512, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    x = DarknetConv(x, filters=1024, kernel_size=3)
    return x_8, x


def Darknet53(x: tf.Tensor) -> Tuple[tf.Tensor]:
    '''YOLOv3
    https://arxiv.org/abs/1804.02767
    '''
    x = DarknetConv(x, filters=32, kernel_size=3)
    x = DarknetBlock(x, filters_1=64, filters_2=32, blocks=1)
    x = DarknetBlock(x, filters_1=128, filters_2=64, blocks=2)
    x = DarknetBlock(x, filters_1=256, filters_2=128, blocks=8)
    x_36 = x
    x = DarknetBlock(x, filters_1=512, filters_2=256, blocks=8)
    x_61 = x
    x = DarknetBlock(x, filters_1=1024, filters_2=512, blocks=4)
    return x_36, x_61, x


def Darknet53_CSPnet(x):
    '''YOLOv4
    https://arxiv.org/abs/2004.10934
    '''
    x = DarknetConv(x, filters=32, kernel_size=3, activation_type='mish')
    route_1 = DarknetConv(
        x, filters=64, kernel_size=3, downsampling=True,
        activation_type='mish'
    )
    route_2 = route_1
    route_2 = DarknetConv(
        route_2, filters=64, kernel_size=1,
        activation_type='mish'
    )
    route_1 = DarknetConv(
        route_1, filters=64, kernel_size=1,
        activation_type='mish'
    )
    route_1 = DarknetResidual(
        route_1, filters_1=64, filters_2=32,
        activation_type='mish'
    )
    route_1 = DarknetConv(
        route_1, filters=64, kernel_size=1,
        activation_type='mish'
    )
    x = tf.concat([route_1, route_2], axis=-1)
    x = DarknetConv(
        x, filters=64, kernel_size=1,
        activation_type='mish'
    )
    x = DarknetBlock_CSPnet(x, filters_1=128, filters_2=64, blocks=2)
    x = DarknetBlock_CSPnet(x, filters_1=256, filters_2=128, blocks=8)
    x_54 = x
    x = DarknetBlock_CSPnet(x, filters_1=512, filters_2=256, blocks=8)
    x_85 = x
    x = DarknetBlock_CSPnet(x, filters_1=1024, filters_2=512, blocks=4)
    x = SPP_Block(x, filters=1024)
    return x_54, x_85, x


def YoloLayer(x: tf.Tensor, filters: int, num_class: int) -> tf.Tensor:
    x = DarknetConv(x, filters=filters, kernel_size=3)
    x = DarknetConv(
        x, filters=3 * (num_class + 5), kernel_size=1,
        activation=False, batch_norm=False
    )
    return x


def YoloBlock(
    x: tf.Tensor,
    filters: int
) -> tf.Tensor:
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    x = DarknetConv(x, filters=filters, kernel_size=3)
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    x = DarknetConv(x, filters=filters, kernel_size=3)
    x = DarknetConv(x, filters=filters//2, kernel_size=1)
    return x


def DetectionLayer(
    outputs: List[tf.Tensor],
    num_class: int,
    channels: int = 3
) -> List[tf.Tensor]:
    rets = list()
    for x in outputs:
        shape = x.shape
        output = tf.reshape(x, (*shape[:3], 3, num_class + 5))
        xywh, conf, prob = tf.split(output, (4, 1, num_class), axis=-1)
        conf = tf.sigmoid(conf)
        prob = tf.sigmoid(prob)
        rets.append(tf.concat([xywh, conf, prob], axis=-1))
    return rets


def YoloV3_tiny(input_layer: tf.Tensor, num_class: int) -> tf.Tensor:
    x_8, x = Darknet53_tiny(input_layer)
    x = DarknetConv(x, filters=256, kernel_size=1)
    large_bbox = YoloLayer(x, filters=512, num_class=num_class)
    x = DarknetConv(x, filters=128, kernel_size=1)
    x = UpSampling(x)
    x = tf.concat([x, x_8], axis=-1)
    middle_bbox = YoloLayer(x, filters=256, num_class=num_class)
    return DetectionLayer([large_bbox, middle_bbox], num_class=num_class)


def YoloV3(input_layer: tf.Tensor, num_class: int) -> tf.Tensor:
    x_36, x_61, x = Darknet53(input_layer)
    x = YoloBlock(x, filters=1024)
    large_bbox = YoloLayer(x, filters=1024, num_class=num_class)
    x = DarknetConv(x, filters=256, kernel_size=1)
    x = UpSampling(x)
    x = tf.concat([x, x_61], axis=-1)
    x = YoloBlock(x, filters=512)
    middle_bbox = YoloLayer(x, filters=512, num_class=num_class)
    x = DarknetConv(x, filters=128, kernel_size=1)
    x = UpSampling(x)
    x = tf.concat([x, x_36], axis=-1)
    x = YoloBlock(x, filters=256)
    small_bbox = YoloLayer(x, filters=256, num_class=num_class)
    return DetectionLayer([
        small_bbox, middle_bbox, large_bbox
    ], num_class=num_class)


def YoloV4(input_layer: tf.Tensor, num_class: int) -> tf.Tensor:
    x_54, x_85, x = Darknet53_CSPnet(input_layer)
    short_cut = x
    x = DarknetConv(x, filters=256, kernel_size=1)
    x = UpSampling(x)
    x_85 = DarknetConv(x_85, filters=256, kernel_size=1)
    x = tf.concat([x_85, x], axis=-1)
    x = YoloBlock(x, filters=512)
    x_85 = x
    x = DarknetConv(x, filters=128, kernel_size=1)
    x = UpSampling(x)
    x_54 = DarknetConv(x_54, filters=128, kernel_size=1)
    x = tf.concat([x_54, x], axis=-1)
    x = YoloBlock(x, filters=256)
    x_54 = x
    small_bbox = YoloLayer(x, filters=256, num_class=num_class)
    x = DarknetConv(x_54, filters=256, kernel_size=3, downsampling=True)
    x = tf.concat([x, x_85], axis=-1)
    x = YoloBlock(x, filters=512)
    x_85 = x
    middle_bbox = YoloLayer(x, filters=512, num_class=num_class)
    x = DarknetConv(x_85, filters=512, kernel_size=3, downsampling=True)
    x = tf.concat([x, short_cut], axis=-1)
    x = YoloBlock(x, filters=1024)
    large_bbox = YoloLayer(x, filters=1024, num_class=num_class)
    return DetectionLayer([
        small_bbox, middle_bbox, large_bbox
    ], num_class=num_class)
