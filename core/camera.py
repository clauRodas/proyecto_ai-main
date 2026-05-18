import cv2
import time
import numpy as np
from typing import Optional


class CameraCapture:
    def __init__(self, cam_id: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._fps = 0.0
        self._t0 = time.perf_counter()
        self._count = 0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        self._count += 1
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= 0.5:
            self._fps = self._count / elapsed
            self._count = 0
            self._t0 = now
        return frame

    @property
    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
