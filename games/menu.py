import time
import cv2
import numpy as np
from typing import Optional, List, Tuple

from games.base_game import BaseGame
from core.renderer import draw_button, draw_text, draw_panel, draw_wrist_cursor, WHITE, YELLOW, CYAN
from utils.landmarks import LEFT_WRIST, RIGHT_WRIST
from utils.math_utils import landmark_to_px

DWELL_SECS = 1.5


class _Button:
    def __init__(self, label: str, action: str, rect: Tuple[int, int, int, int]):
        self.label  = label
        self.action = action
        self.rect   = rect
        self._t0: Optional[float] = None

    def update(self, wrist_positions: List[Tuple[int, int]]) -> float:
        """Returns dwell ratio 0–1. Handles multiple wrists with OR logic."""
        x, y, w, h = self.rect
        hovering = any(x <= p[0] <= x + w and y <= p[1] <= y + h for p in wrist_positions)
        if hovering:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            return min((time.perf_counter() - self._t0) / DWELL_SECS, 1.0)
        self._t0 = None
        return 0.0

    def reset(self) -> None:
        self._t0 = None


class MainMenu(BaseGame):
    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._w = frame_w
        self._h = frame_h
        self._next: Optional[str] = None
        self._buttons: List[_Button] = []
        self._ratios: List[float] = []
        self._wrists: List[Optional[Tuple[int, int]]] = [None, None]
        self._build()

    def _build(self) -> None:
        bw, bh = 300, 78
        cx = self._w // 2 - bw // 2
        specs = [
            ("BOXEO",  "boxing",         self._h // 2 - 95),
            ("POSES",  "pose_challenge",  self._h // 2 + 5),
            ("SALIR",  "exit",            self._h // 2 + 105),
        ]
        for label, action, top_y in specs:
            self._buttons.append(_Button(label, action, (cx, top_y, bw, bh)))
        self._ratios = [0.0] * len(self._buttons)

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, landmarks: Optional[list],
               frame_w: int, frame_h: int) -> None:
        self._wrists = [None, None]
        wrist_pts: List[Tuple[int, int]] = []

        if landmarks:
            lw = landmark_to_px(landmarks[LEFT_WRIST],  frame_w, frame_h)
            rw = landmark_to_px(landmarks[RIGHT_WRIST], frame_w, frame_h)
            self._wrists = [lw, rw]
            wrist_pts    = [lw, rw]

        for i, btn in enumerate(self._buttons):
            ratio = btn.update(wrist_pts)
            self._ratios[i] = ratio
            if ratio >= 1.0:
                self._next = btn.action
                return

        if not landmarks:
            for btn in self._buttons:
                btn.reset()

    def render(self, frame: np.ndarray) -> None:
        draw_panel(frame, (0, 0, self._w, self._h), color=(5, 5, 30), alpha=0.45)

        title = "CONSOLA MULTIJUEGOS"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        draw_text(frame, title, (self._w // 2 - tw // 2, 70),
                  scale=1.1, color=YELLOW, thickness=3)

        hint = "Acerca una muneca al boton para seleccionar"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        draw_text(frame, hint, (self._w // 2 - hw // 2, 105),
                  scale=0.5, color=WHITE, thickness=1)

        for i, btn in enumerate(self._buttons):
            draw_button(frame, btn.rect, btn.label, hover_ratio=self._ratios[i])

        for pt in self._wrists:
            draw_wrist_cursor(frame, pt, color=CYAN)

    def reset(self) -> None:
        self._next = None
        for btn in self._buttons:
            btn.reset()
        self._ratios = [0.0] * len(self._buttons)
        self._wrists = [None, None]

    @property
    def next_state(self) -> Optional[str]:
        return self._next

    @property
    def name(self) -> str:
        return "menu"
