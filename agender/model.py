#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    AveragePooling2D,
    BatchNormalization,
    Dense,
    Activation,
    Add,
    Flatten
)
from tensorflow.keras.regularizers import l2


def WideConv(x: tf.Tensor, filters: int, ksize: int, stride: int) -> tf.Tensor:
    return Conv2D(
        filters=filters, kernel_size=(ksize, ksize),
        strides=(stride, stride), padding='SAME',
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0005),
        use_bias=False
    )(x)


def ResidualBlock_1(x: tf.Tensor, filters: int, stride: int) -> tf.Tensor:
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    short_cut = x
    x = WideConv(x, filters=filters, ksize=3, stride=stride)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = WideConv(x, filters=filters, ksize=3, stride=1)
    short_cut = WideConv(short_cut, filters=filters, ksize=1, stride=stride)
    return Add()([x, short_cut])


def ResidualBlock_2(x: tf.Tensor, filters: int) -> tf.Tensor:
    short_cut = x
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = WideConv(x, filters=filters, ksize=3, stride=1)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = WideConv(x, filters=filters, ksize=3, stride=1)
    return Add()([x, short_cut])


def WideLayer(
    x: tf.Tensor,
    filters: int,
    stride: int,
    n_blocks: int
) -> tf.Tensor:
    x = ResidualBlock_1(x, filters=filters, stride=stride)
    for _ in range(1, n_blocks):
        x = ResidualBlock_2(x, filters=filters)
    return x


def WideResNet(
    input_layer: tf.Tensor,
    depth: int = 16,
    k: int = 8
) -> tf.Tensor:
    n_blocks = (depth - 4) // 6
    print(input_layer.shape)
    x = WideConv(input_layer, filters=16, ksize=3, stride=1)
    x = WideLayer(x, filters=16 * k, stride=1, n_blocks=n_blocks)
    x = WideLayer(x, filters=32 * k, stride=2, n_blocks=n_blocks)
    x = WideLayer(x, filters=64 * k, stride=2, n_blocks=n_blocks)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='SAME')(x)
    x = Flatten()(x)
    pred_gender = Dense(
        units=2,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0005),
        use_bias=False,
        activation='softmax',
        name='pred_gender'
    )(x)
    pred_age = Dense(
        units=101,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0005),
        use_bias=False,
        activation='softmax',
        name='pred_age'
    )(x)
    return [pred_gender, pred_age]
