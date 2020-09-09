#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List
from PIL import Image
import cv2
import numpy as np
from decode import non_maximum_supression

# mode of selective search segmentation
# 'single' or 'fast' or 'quality'
SELECTIVE_MODE = 'single'
# minimum are of segmented rectagle
MIN_RECT_AREA = 4000
# maximun number of selected rectangle
MAX_RECTS = 3


def search_selective(current: Image) -> List:
    curr = cv2.cvtColor(
        np.array(current, dtype=np.uint8),
        cv2.COLOR_RGB2BGR
    )
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(curr)
    if SELECTIVE_MODE == 'single':
        ss.switchToSingleStrategy()
    elif SELECTIVE_MODE == 'fast':
        ss.switchToSelectiveSearchFast()
    elif SELECTIVE_MODE == 'quality':
        ss.switchToSelectiveSearchQUality()
    else:
        raise ValueError('SELECTIVE_MODE is invalid')
    rects = ss.process()
    bboxes = list()
    for rect in rects:
        x, y, w, h = rect
        area = w * h
        if area < MIN_RECT_AREA:
            continue
        bboxes.append(rect)
        if len(bboxes) >= MAX_RECTS:
            break
    return bboxes


def select_selective(objects: List) -> List:
    if len(objects) == 0:
        return []
    id2label = dict()
    bboxes = list()
    for obj in objects:
        id2label[obj['index']] = obj['name']
        bbox = list(obj['bbox'])
        bbox.append(obj['index'])
        bbox.append(obj['prob'])
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    best_bboxes = non_maximum_supression(bboxes=bboxes)
    ret = list()
    for bbox in best_bboxes:
        ret.append({
            'name': id2label[bbox[4]],
            'prob': bbox[5],
            'bbox': tuple(bbox[:4]),
        })
    return ret
