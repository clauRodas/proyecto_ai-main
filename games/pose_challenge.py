import time
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

from games.base_game import BaseGame
from core.renderer import (
    draw_text, draw_progress_bar, draw_panel,
    WHITE, GREEN, RED, YELLOW, CYAN, ORANGE,
)
from utils.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW,    RIGHT_ELBOW,
    LEFT_WRIST,    RIGHT_WRIST,
    LEFT_HIP,      RIGHT_HIP,
    LEFT_KNEE,     RIGHT_KNEE,
    LEFT_ANKLE,    RIGHT_ANKLE,
)
from utils.math_utils import calc_angle

HOLD_SECS   = 2.0    # seconds to hold a matching pose
THRESHOLD   = 72.0   # minimum similarity % to start the hold timer
TOLERANCE   = 25.0   # degrees tolerance per joint
SUCCESS_LINGER = 1.8 # seconds to show "EXCELENTE" screen


# Reference poses: {joint_key: target_angle_degrees}
# Angles are at the middle joint of each triplet.
# left_elbow  = angle at elbow:    shoulder → elbow → wrist
# left_shoulder = angle at shoulder: hip → shoulder → elbow
# left_knee   = angle at knee:     hip → knee → ankle
REFERENCE_POSES: List[Dict] = [
    {
        "name": "T-POSE",
        "description": "Extiende los brazos horizontalmente",
        "angles": {
            "left_elbow":    170,
            "right_elbow":   170,
            "left_shoulder":  90,
            "right_shoulder": 90,
        },
    },
    {
        "name": "BRAZOS ARRIBA",
        "description": "Levanta ambos brazos sobre la cabeza",
        "angles": {
            "left_shoulder":  160,
            "right_shoulder": 160,
            "left_elbow":     165,
            "right_elbow":    165,
        },
    },
    {
        "name": "VICTORIA",
        "description": "Forma una V con los brazos",
        "angles": {
            "left_shoulder":  130,
            "right_shoulder": 130,
            "left_elbow":     165,
            "right_elbow":    165,
        },
    },
    {
        "name": "MUSCULOSO",
        "description": "Dobla los codos, brazos al costado",
        "angles": {
            "left_shoulder":  90,
            "right_shoulder": 90,
            "left_elbow":     90,
            "right_elbow":    90,
        },
    },
    {
        "name": "BAILE",
        "description": "Brazo izquierdo arriba, derecho al lado",
        "angles": {
            "left_shoulder":  155,
            "right_shoulder":  55,
            "left_elbow":     165,
            "right_elbow":    165,
        },
    },
]


def _extract_angles(landmarks: list, fw: int, fh: int) -> Dict[str, float]:
    def pt(idx: int) -> Tuple[float, float]:
        lm = landmarks[idx]
        return (lm.x * fw, lm.y * fh)

    try:
        return {
            "left_elbow":    calc_angle(pt(LEFT_SHOULDER),  pt(LEFT_ELBOW),   pt(LEFT_WRIST)),
            "right_elbow":   calc_angle(pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW),  pt(RIGHT_WRIST)),
            "left_shoulder": calc_angle(pt(LEFT_HIP),       pt(LEFT_SHOULDER),  pt(LEFT_ELBOW)),
            "right_shoulder":calc_angle(pt(RIGHT_HIP),      pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW)),
            "left_knee":     calc_angle(pt(LEFT_HIP),       pt(LEFT_KNEE),    pt(LEFT_ANKLE)),
            "right_knee":    calc_angle(pt(RIGHT_HIP),      pt(RIGHT_KNEE),   pt(RIGHT_ANKLE)),
        }
    except Exception:
        return {}


def _similarity(user: Dict[str, float], ref: Dict[str, float]) -> float:
    scores = []
    for joint, target in ref.items():
        val = user.get(joint)
        if val is not None:
            scores.append(max(0.0, 100.0 - abs(val - target) / TOLERANCE * 100.0))
    return sum(scores) / len(scores) if scores else 0.0


class PoseChallenge(BaseGame):
    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._w = frame_w
        self._h = frame_h
        self._next: Optional[str] = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._next        = None
        self._idx         = 0
        self._score       = 0
        self._sim         = 0.0
        self._hold_start: Optional[float] = None
        self._hold_ratio  = 0.0
        self._user_angles: Dict[str, float] = {}
        self._success     = False
        self._success_t   = 0.0

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, landmarks: Optional[list],
               frame_w: int, frame_h: int) -> None:
        now = time.perf_counter()

        if self._idx >= len(REFERENCE_POSES):
            self._next = "menu"
            return

        if self._success:
            if now - self._success_t >= SUCCESS_LINGER:
                self._success = False
                self._idx    += 1
                if self._idx >= len(REFERENCE_POSES):
                    self._next = "menu"
            return

        if landmarks:
            self._user_angles = _extract_angles(landmarks, frame_w, frame_h)
            ref = REFERENCE_POSES[self._idx]["angles"]
            self._sim = _similarity(self._user_angles, ref)

            if self._sim >= THRESHOLD:
                if self._hold_start is None:
                    self._hold_start = now
                self._hold_ratio = min((now - self._hold_start) / HOLD_SECS, 1.0)
                if self._hold_ratio >= 1.0:
                    self._score     += int(self._sim)
                    self._success    = True
                    self._success_t  = now
                    self._hold_start = None
                    self._hold_ratio = 0.0
            else:
                self._hold_start = None
                self._hold_ratio = 0.0
        else:
            self._user_angles = {}
            self._sim         = 0.0
            self._hold_start  = None
            self._hold_ratio  = 0.0

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray) -> None:
        if self._idx >= len(REFERENCE_POSES):
            return

        # Success flash
        if self._success:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self._w, self._h), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            draw_text(frame, "EXCELENTE!",
                      (self._w // 2 - 115, self._h // 2 - 10),
                      scale=1.6, color=(0, 255, 0), thickness=4)
            draw_text(frame, f"+{int(self._sim)} pts",
                      (self._w // 2 - 60, self._h // 2 + 50),
                      scale=1.0, color=YELLOW, thickness=2)
            return

        pose = REFERENCE_POSES[self._idx]

        # Sidebar panel
        panel_w = 210
        draw_panel(frame, (0, 0, panel_w, self._h), color=(10, 10, 40), alpha=0.70)

        draw_text(frame, "IMITA LA POSE", (8, 30),  scale=0.65, color=YELLOW, thickness=2)
        draw_text(frame, pose["name"],    (8, 60),  scale=0.75, color=WHITE,  thickness=2)

        # Word-wrap description naively
        desc = pose["description"]
        draw_text(frame, desc[:25], (8, 85),  scale=0.45, color=(200, 200, 200), thickness=1)
        if len(desc) > 25:
            draw_text(frame, desc[25:], (8, 103), scale=0.45, color=(200, 200, 200), thickness=1)

        # Per-joint feedback
        ref_angles = pose["angles"]
        y_off = 125
        joint_labels = {
            "left_elbow":     "Codo izq",
            "right_elbow":    "Codo der",
            "left_shoulder":  "Hombro izq",
            "right_shoulder": "Hombro der",
            "left_knee":      "Rodilla izq",
            "right_knee":     "Rodilla der",
        }
        for joint, target in ref_angles.items():
            val = self._user_angles.get(joint)
            label = joint_labels.get(joint, joint)
            if val is not None:
                diff = abs(val - target)
                ok   = diff <= TOLERANCE
                color = GREEN if ok else (0, 165, 255) if diff < TOLERANCE * 2 else RED
                draw_text(frame, f"{label}: {val:.0f}({target:.0f})",
                          (8, y_off), scale=0.42, color=color, thickness=1)
            else:
                draw_text(frame, f"{label}: --",
                          (8, y_off), scale=0.42, color=(120, 120, 120), thickness=1)
            y_off += 22

        # Similarity bar
        sim_color = GREEN if self._sim >= THRESHOLD else ORANGE if self._sim >= 50 else RED
        draw_text(frame, f"SIMILITUD: {self._sim:.0f}%",
                  (8, self._h - 95), scale=0.6, color=sim_color, thickness=2)
        draw_progress_bar(frame, (8, self._h - 75), (panel_w - 16, 16),
                          self._sim, 100.0, fg_color=sim_color)

        # Hold bar
        if self._hold_ratio > 0:
            draw_text(frame, "MANTEN LA POSE!",
                      (8, self._h - 50), scale=0.55, color=CYAN, thickness=1)
            draw_progress_bar(frame, (8, self._h - 32), (panel_w - 16, 12),
                              self._hold_ratio, 1.0, fg_color=CYAN)

        # Top-right counter
        counter_txt = f"Pose {self._idx + 1}/{len(REFERENCE_POSES)}"
        (cw, _), _ = cv2.getTextSize(counter_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        draw_panel(frame, (self._w - cw - 20, 0, cw + 20, 65), color=(10, 10, 40), alpha=0.65)
        draw_text(frame, counter_txt,         (self._w - cw - 10, 28), scale=0.6,  color=WHITE)
        draw_text(frame, f"Pts: {self._score}", (self._w - cw - 10, 55), scale=0.55, color=YELLOW)

    @property
    def next_state(self) -> Optional[str]:
        return self._next

    @property
    def name(self) -> str:
        return "pose_challenge"
