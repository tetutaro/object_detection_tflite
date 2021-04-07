#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
import time
from src.config import Config
from src.camera import get_camera
from src.detection_model import get_detection_model
from src.prediction_model import get_prediction_model
from src.motion import Motion
from src.selective import Selective


class Detector(object):
    def __init__(self: Detector, config: Config) -> None:
        self.config = config
        self.camera = get_camera(config=config)
        self.model = get_detection_model(config=config)
        return

    def run(self: Detector) -> None:
        self.camera.start()
        try:
            framecount = 0
            for image in self.camera.yield_image():
                framecount += 1
                if framecount >= self.config.fastforward:
                    start_time = time.perf_counter()
                    objects = self.model.inference(image)
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    self.camera.clear()
                    self.camera.draw_objects(objects)
                    self.camera.draw_time(elapsed_ms)
                    self.camera.draw_count(len(objects))
                    self.camera.update()
                    framecount = 0
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
        return


class DetectorAgender(object):
    def __init__(
        self: DetectorAgender,
        config_detect: Config,
        config_predict: Config
    ) -> None:
        self.config = config_detect
        self.camera = get_camera(config=config_detect)
        self.model_detect = get_detection_model(config=config_detect)
        self.model_predict = get_prediction_model(config=config_predict)
        return

    def run(self: DetectorAgender) -> None:
        self.camera.start()
        try:
            framecount = 0
            for image in self.camera.yield_image():
                framecount += 1
                if framecount >= self.config.fastforward:
                    start_time = time.perf_counter()
                    objects = self.model_detect.inference(image)
                    self.model_predict.predict(image, objects)
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    self.camera.clear()
                    self.camera.draw_objects(objects)
                    self.camera.draw_time(elapsed_ms)
                    self.camera.draw_count(len(objects))
                    self.camera.update()
                    framecount = 0
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()


class DetectorMotion(object):
    def __init__(
        self: DetectorMotion,
        config: Config,
    ) -> None:
        self.config = config
        self.camera = get_camera(config=config)
        self.model = get_prediction_model(config=config)
        self.motion = Motion()
        return

    def run(self: DetectorMotion) -> None:
        self.camera.start()
        try:
            framecount = 0
            for image in self.camera.yield_image():
                framecount += 1
                if framecount >= self.config.fastforward:
                    start_time = time.perf_counter()
                    objects = self.motion.detect(image)
                    objects = self.model.predict(image, objects)
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    self.camera.clear()
                    self.camera.draw_objects(objects)
                    self.camera.draw_time(elapsed_ms)
                    self.camera.draw_count(len(objects))
                    self.camera.update()
                    framecount = 0
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
        return


class DetectorSelective(object):
    def __init__(
        self: DetectorSelective,
        config: Config,
    ) -> None:
        self.config = config
        self.camera = get_camera(config=config)
        self.model = get_prediction_model(config=config)
        self.selective = Selective(
            config=config, id2label=self.model.id2label
        )
        return

    def run(self: DetectorSelective) -> None:
        self.camera.start()
        try:
            framecount = 0
            for image in self.camera.yield_image():
                framecount += 1
                if framecount >= self.config.fastforward:
                    start_time = time.perf_counter()
                    objects = self.selective.search(image)
                    objects = self.model.predict(image, objects)
                    objects = self.selective.select(objects)
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    self.camera.clear()
                    self.camera.draw_objects(objects)
                    self.camera.draw_time(elapsed_ms)
                    self.camera.draw_count(len(objects))
                    self.camera.update()
                    framecount = 0
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
