#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np

# the weighted average weight of current and past frames
WEIGHT_ACCUMULATE = 0.7
# the threshold for detecting dot movement
DOT_THRESHOLD = 5
# the minimun area of contour
MIN_CONTOUR_AREA = 4000


class Motion(object):
    def __init__(self: Motion) -> None:
        self.before = None
        return

    def detect(self: Motion, current: Image) -> List[Dict]:
        curr = cv2.cvtColor(
            np.array(current, dtype=np.float32),
            cv2.COLOR_RGB2GRAY
        )
        if self.before is None:
            self.before = curr.copy()
            return []
        base = self.before
        self.before = curr.copy()
        cv2.accumulateWeighted(curr, base, WEIGHT_ACCUMULATE)
        diff = cv2.absdiff(
            cv2.convertScaleAbs(curr),
            cv2.convertScaleAbs(base)
        )
        thresh = cv2.threshold(
            diff, DOT_THRESHOLD, 255, cv2.THRESH_BINARY
        )[1]
        # findCounters() returns...
        # contours, hierarchy (Open CV 2) (imutils.is_cv2() is True)
        # image, contours, hierarchy (Open CV 3) (imutils.is_cv3() is True)
        # contours, hierarchy (Open CV 4) (imutils.is_cv4() is True)
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        objects = list()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            objects.append({
                'bbox': (x, y, x + w, y + h),
            })
        return objects
