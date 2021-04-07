#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np
from src.config import Config
from src.utils import filter_bboxes

# mode of selective search segmentation
# 'single' or 'fast' or 'quality'
SELECTIVE_MODE = 'fast'
# minimum are of segmented rectagle
MIN_RECT_AREA = 4000
# maximun number of selected rectangle
MAX_RECTS = 3


class Selective(object):
    def __init__(self: Selective, config: Config, id2label: Dict) -> None:
        self.config = config
        self.id2label = id2label
        return

    def search(self: Selective, current: Image) -> List[Dict]:
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
        objects = list()
        for rect in rects:
            x, y, w, h = rect
            area = w * h
            if area < MIN_RECT_AREA:
                continue
            objects.append({
                'bbox': (x, y, x + w, y + h),
            })
            if len(objects) >= MAX_RECTS:
                break
        return objects

    def select(self: Selective, objects: List[Dict]) -> List[Dict]:
        if len(objects) == 0:
            return []
        bboxes = list()
        for obj in objects:
            bboxes.append((*obj['bbox'], obj['category_id'], obj['prob']))
        bboxes = np.array(bboxes)
        bboxes = filter_bboxes(
            bboxes=bboxes,
            conf_threshold=self.config.prob_threshold,
            iou_threshold=self.config.iou_thrreshold
        )
        ret = list()
        for bbox in bboxes.tolist():
            if bbox[5] >= self.config.prob_threshold:
                ret.append({
                    'name': self.id2label[bbox[4]],
                    'prob': bbox[5],
                    'bbox': tuple(bbox[:4]),
                })
        return ret
