#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional


def _round_up(value: int, n: int) -> int:
    return n * ((value + (n - 1)) // n)


class Config(object):
    def __init__(
        self: Config,
        media: Optional[str],
        height: int,
        width: int,
        hflip: bool,
        vflip: bool,
        model: str,
        quant: str,
        target: str,
        fontsize: int,
        fastforward: int,
        iou_threshold: Optional[float] = 0.45,
        conf_threshold: Optional[float] = 0.5,
        prob_threshold: Optional[float] = 0.5
    ) -> None:
        self.media = media
        self.camera_height = _round_up(height, 16)
        self.camera_width = _round_up(width, 32)
        self.hflip = hflip
        self.vflip = vflip
        self.model = model
        self.quant = quant
        if self.quant in ['int8', 'tpu']:
            self.is_int8 = True
        else:
            self.is_int8 = False
        self.target = target
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.prob_threshold = prob_threshold
        self.fontsize = fontsize
        if fastforward > 1:
            self.fastforward = fastforward
        else:
            self.fastforward = 1
        return
