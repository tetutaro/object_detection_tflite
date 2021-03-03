#!/usr/bin/env python
# -*- coding:utf-8 -*-
from convert_tflite import test_tflite


if __name__ == '__main__':
    # test tflite
    for model in ['s', 'm', 'l', 'x']:
        for mode in ['fp32', 'fp16', 'int8']:
            test_tflite(model=f'yolov5{model}', mode=mode)
