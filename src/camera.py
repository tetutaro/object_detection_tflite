#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Generator, Optional
import os
import io
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import numpy as np
import cv2
import platform
if platform.system() == 'Linux':  # RaspberryPi
    import picamera
from src.config import Config

FRAME_PER_SECOND = 30


class Camera(object):
    def __init__(self: Camera, config: Config) -> None:
        self.config = config
        self._buffer = Image.new(
            'RGBA', (self.config.camera_width, self.config.camera_height)
        )
        self._overlay = None
        self._draw = ImageDraw.Draw(self._buffer)
        self._default_color = (0xff, 0xff, 0xff, 0xff)
        self._font = ImageFont.truetype(
            font='TakaoGothic.ttf', size=config.fontsize
        )
        return

    def clear(self: Camera) -> None:
        self._draw.rectangle(
            (0, 0, self.config.camera_width, self.config.camera_height),
            fill=(0, 0, 0, 0x00)
        )
        return

    def draw_objects(self: Camera, objects: List) -> None:
        for obj in objects:
            self._draw_object(obj)
        return

    def draw_time(self: Camera, elapsed_ms: float) -> None:
        text = 'Elapsed Time: %.1f[ms]' % elapsed_ms
        if self.config.fastforward > 1:
            text += ' (speed x%d)' % self.config.fastforward
        self._draw_text(
            text, location=(5, 5), color=None
        )
        return

    def draw_count(self: Camera, count: int) -> None:
        self._draw_text(
            'Detected Objects: %d' % count,
            location=(5, 5 + self.config.fontsize), color=None
        )
        return

    def _draw_object(self: Camera, object: Dict) -> None:
        prob = object['prob']
        color = tuple(np.array(np.array(cm.jet((
            prob - self.config.conf_threshold
        ) / (
            1.0 - self.config.conf_threshold
        ))) * 255, dtype=np.uint8).tolist())
        self._draw_box(
            rect=object['bbox'], color=color
        )
        name = object.get('name')
        xoff = object['bbox'][0] + 5
        yoff = object['bbox'][1] + 5
        if name is not None:
            self._draw_text(
                name, location=(xoff, yoff), color=color
            )
            yoff += self.config.fontsize
        self._draw_text(
            '%.3f' % prob, location=(xoff, yoff), color=color
        )
        return

    def _draw_box(
        self: Camera,
        rect: Tuple[int],
        color: Optional[Tuple[int]]
    ) -> None:
        outline = color or self._default_color
        self._draw.rectangle(rect, fill=None, outline=outline)
        return

    def _draw_text(
        self: Camera,
        text: str,
        location: Tuple[int, int],
        color: Optional[Tuple[int, int, int, int]]
    ) -> None:
        color = color or self._default_color
        self._draw.text(location, text, fill=color, font=self._font)
        return

    def update(self: Camera) -> None:
        if self._overlay is not None:
            self._camera.remove_overlay(self._overlay)
        if self._buffer is None:
            return
        self._overlay = self._camera.add_overlay(
            self._buffer.tobytes(),
            format='rgba', layer=3,
            size=(self.config.camera_width, self.config.camera_height)
        )
        self._overlay.update(self._buffer.tobytes())
        return


class RaspiCamera(Camera):
    def __init__(self: RaspiCamera, config: Config) -> None:
        super().__init__(config=config)
        self._camera = picamera.PiCamera(
            resolution=(
                self.config.camera_width, self.config.camera_height
            ),
            framerate=FRAME_PER_SECOND
        )
        self._camera.hflip = self.config.hflip
        self._camera.vflip = self.config.vflip
        return

    def start(self: RaspiCamera) -> None:
        self._camera.start_preview()
        return

    def yield_image(self: RaspiCamera) -> Generator[Image, None]:
        self._stream = io.BytesIO()
        for _ in self._camera.capture_continuous(
            self._stream,
            format='jpeg',
            use_video_port=True
        ):
            self._stream.seek(0)
            image = Image.open(self._stream).convert('RGB')
            yield image
        return

    def update(self: RaspiCamera) -> None:
        super().update()
        self._stream.seek(0)
        self._stream.truncate()
        return

    def stop(self: RaspiCamera) -> None:
        self._camera.stop_preview()
        return


class VideoCamera(Camera):
    def __init__(self: VideoCamera, config: Config) -> None:
        if config.media is None:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                float(config.camera_height)
            )
            self._camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                float(config.camera_width)
            )
        else:
            self._camera = cv2.VideoCapture(config.media)
        # adjust aspect ratio
        config.camera_height = int(self._camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        ))
        config.camera_width = int(self._camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        ))
        super().__init__(config=config)
        # set flipcode
        if config.hflip:
            if config.vflip:
                self.flipcode = -1
            else:
                self.flipcode = 1
        elif config.vflip:
            self.flipcode = 0
        else:
            self.flipcode = None
        return

    def start(self: VideoCamera) -> None:
        self.window = 'Object Detection'
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(
            self.window,
            self.config.camera_width, self.config.camera_height
        )
        return

    def yield_image(self: VideoCamera) -> Generator[Image, None]:
        while True:
            _, image = self._camera.read()
            if image is None:
                time.sleep(1)
                continue
            if self.flipcode is not None:
                image = cv2.flip(image, self.flipcode)
            self.image = image
            yield Image.fromarray(image.copy()[..., ::-1])
        return

    def update(self: VideoCamera) -> None:
        overlay = np.array(self._buffer, dtype=np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        image = cv2.addWeighted(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
            overlay, 0.5, 2.2
        )
        cv2.imshow(self.window, image)
        key = cv2.waitKey(1000 // FRAME_PER_SECOND)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: VideoCamera) -> None:
        cv2.destroyAllWindows()
        self._camera.release()
        return


class ImageCamera(Camera):
    def __init__(self: ImageCamera, config: Config) -> None:
        self.image = cv2.imread(config.media)
        config.camera_height = self.image.shape[0]
        config.camera_width = self.image.shape[1]
        config.fastforward = 1
        super().__init__(config=config)
        return

    def start(self: ImageCamera) -> None:
        self.window = 'Object Detection'
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(
            self.window,
            self.config.camera_width, self.config.camera_height
        )
        return

    def yield_image(self: ImageCamera) -> Generator[Image, None]:
        image = self.image.copy()[..., ::-1]
        yield Image.fromarray(image)
        return

    def update(self: ImageCamera) -> None:
        overlay = np.array(self._buffer, dtype=np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        image = cv2.addWeighted(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA), 0.5,
            overlay, 0.5, 2.2
        )
        cv2.imshow(self.window, image)
        key = cv2.waitKey(0)
        if key == 99:
            raise KeyboardInterrupt
        return

    def stop(self: ImageCamera) -> None:
        cv2.destroyAllWindows()
        return


def get_camera(config: Config) -> Camera:
    if config.media is not None:
        if not os.path.exists(config.media):
            raise ValueError('set existed file')
        ext = os.path.splitext(config.media)[1]
        if ext in ['.jpg', '.png']:
            camera = ImageCamera(config=config)
        else:
            camera = VideoCamera(config=config)
    elif platform.system() == 'Linux':  # RaspberryPi
        camera = RaspiCamera(config=config)
    elif platform.system() == 'Darwin':  # MacOS
        camera = VideoCamera(config=config)
    else:
        raise NotImplementedError()
    return camera
