#!/usr/bin/env python
# -*- coding:utf-8 -*-
from convert_tflite import test_tflite


if __name__ == '__main__':
    # test tflite
    for model in ['yolov3-tiny', 'yolov3', 'yolov4']:
        for mode in ['fp32', 'fp16', 'int8']:
            test_tflite(model=model, mode=mode)
