import os
import time
import random
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict

from games.base_game import BaseGame
from core.renderer import (
    draw_target, draw_text, draw_progress_bar, draw_panel, draw_rounded_panel,
    draw_wrist_cursor, load_sprite, overlay_image_alpha, WHITE, RED, GREEN,
    YELLOW, CYAN, ORANGE,
)
from utils.landmarks import LEFT_WRIST, RIGHT_WRIST, POSE_CONNECTIONS
from utils.math_utils import landmark_to_px, distance_2d

GAME_DURATION   = 60.0
TARGET_LIFETIME = 3.0
HIT_SLOP        = 22          # extra pixels around target radius that count as hit
COMBO_WINDOW    = 1.5         # seconds between hits to keep combo alive
GAMEOV_LINGER   = 2.5         # seconds before returning to menu after game-over


class _Target:
    COLORS = [
        (0,   0,   220),   # red
        (0,   130, 255),   # orange
        (200, 0,   160),   # magenta
        (0,   80,  200),   # dark-orange
    ]

    def __init__(self, cx: int, cy: int, radius: int):
        self.center    = (cx, cy)
        self.radius    = radius
        self.color     = random.choice(self.COLORS)
        self._born     = time.perf_counter()
        self.hit       = False
        self.hit_time: Optional[float] = None

    @property
    def alive(self) -> bool:
        return not self.hit and (time.perf_counter() - self._born) < TARGET_LIFETIME

    @property
    def age_ratio(self) -> float:
        return min((time.perf_counter() - self._born) / TARGET_LIFETIME, 1.0)


class BoxingGame(BaseGame):
    TRAINER_DIR = os.path.join("assets", "boxing_sprites")
    CANVAS_W = 1280
    CANVAS_H = 720
    GAME_AREA = (40, 90, 850, 540)
    CAM_AREA = (930, 370, 300, 225)
    SEQUENCE = [
        ("punch_left", 2.5),
        ("punch_right", 2.5),
        ("rest", 1.8),
        ("punch_left", 2.5),
        ("punch_right", 2.5),
        ("punch_left", 2.5),
        ("rest", 1.8),
    ]
    # Intercambia la muñeca izquierda y derecha de MediaPipe porque la cámara
    # está espejada: lo que MediaPipe detecta como "izquierda" es la derecha
    # física del jugador frente al espejo, y viceversa.
    MIRROR_HANDS = True

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._w = frame_w
        self._h = frame_h
        self._sprites = self._load_trainer_sprites()
        self._movement_frames = ["idle", "move_left", "idle", "move_right"]
        self._sequence_index = 0
        self._action_start = time.perf_counter()
        self.reset()

    def _load_trainer_sprites(self) -> Dict[str, np.ndarray]:
        sprites = {}
        names = ["idle", "punch_left", "punch_right", "move_left", "move_right"]
        for name in names:
            path = os.path.join(self.TRAINER_DIR, f"{name}.png")
            sprites[name] = load_sprite(path, size=(520, 520))
        return sprites

    def _set_action(self, index: int) -> None:
        self._sequence_index = index % len(self.SEQUENCE)
        self._current_action, self._action_timeout = self.SEQUENCE[self._sequence_index]
        self._action_start = time.perf_counter()
        self._required_hand = (
            "left" if self._current_action == "punch_left" else
            "right" if self._current_action == "punch_right" else None
        )
        self._current_target = self._create_target_for_action(self._current_action)
        self._feedback_msg = ""
        self._feedback_time = 0.0

    def _create_target_for_action(self, action: str) -> Optional[_Target]:
        # The camera frame is mirrored for display, so the visible left punch
        # target is on the right side of the image and vice versa.
        if action == "punch_left":
            target = _Target(460, 260, 52)
            target.color = (0, 0, 220)
            return target
        if action == "punch_right":
            target = _Target(180, 260, 52)
            target.color = (0, 130, 255)
            return target
        return None

    def _current_trainer_sprite(self) -> np.ndarray:
        if self._current_action == "punch_left":
            return self._sprites.get("punch_left", self._sprites["idle"])
        if self._current_action == "punch_right":
            return self._sprites.get("punch_right", self._sprites["idle"])
        move_phase = int((time.perf_counter() - self._action_start) / 0.45) % len(self._movement_frames)
        return self._sprites.get(self._movement_frames[move_phase], self._sprites["idle"])

    def _map_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x0, y0, w0, h0 = self.GAME_AREA
        sx = w0 / self._w
        sy = h0 / self._h
        return (int(x0 + point[0] * sx), int(y0 + point[1] * sy))

    def _map_radius(self, radius: int) -> int:
        x0, y0, w0, h0 = self.GAME_AREA
        scale = min(w0 / self._w, h0 / self._h)
        return max(4, int(radius * scale))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._next        = None
        self._score       = 0
        self._lives       = 3
        self._combo       = 0
        self._targets:    List[_Target] = []
        self._effects:    List[Tuple[int, int, float]] = []   # (x, y, born)
        self._start       = time.perf_counter()
        self._last_hit    = 0.0
        self._wrists: List[Optional[Tuple[int, int]]] = [None, None]
        self._landmarks   = None
        self._game_over   = False
        self._over_time   = 0.0
        self._feedback_msg = ""
        self._feedback_color = WHITE
        self._feedback_time = 0.0
        self._set_action(0)

    def _set_feedback(self, message: str, color: Tuple[int, int, int]) -> None:
        self._feedback_msg = message
        self._feedback_color = color
        self._feedback_time = time.perf_counter()

    def _advance_action(self, failed: bool = False) -> None:
        if failed:
            self._combo = 0
            self._lives = max(0, self._lives - 1)
            self._set_feedback("FALLASTE", RED)
        self._set_action(self._sequence_index + 1)

    def _process_punch(self, now: float) -> None:
        if self._current_target is None:
            return

        correct_wrist = self._wrists[0] if self._required_hand == "left" else self._wrists[1]
        wrong_wrist = self._wrists[1] if self._required_hand == "left" else self._wrists[0]
        hit_radius = self._current_target.radius + HIT_SLOP
        wrong_radius = self._current_target.radius + int(HIT_SLOP * 0.75)

        correct_dist = distance_2d(correct_wrist, self._current_target.center) if correct_wrist is not None else float("inf")
        wrong_dist = distance_2d(wrong_wrist, self._current_target.center) if wrong_wrist is not None else float("inf")

        if correct_dist <= hit_radius:
            self._current_target.hit = True
            self._current_target.hit_time = now
            self._combo = self._combo + 1 if now - self._last_hit < COMBO_WINDOW else 1
            self._last_hit = now
            self._score += 15 * self._combo
            self._effects.append((*self._current_target.center, now))
            self._set_feedback("PERFECTO", CYAN if self._combo > 1 else GREEN)
            self._advance_action(failed=False)
            return

        if wrong_dist <= wrong_radius and wrong_dist < correct_dist:
            self._set_feedback("MANO INCORRECTA", ORANGE)
            return

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, landmarks: Optional[list],
               frame_w: int, frame_h: int) -> None:
        now = time.perf_counter()
        self._landmarks = landmarks

        if self._game_over:
            if now - self._over_time >= GAMEOV_LINGER:
                self._next = "menu"
            return

        elapsed = now - self._start
        if elapsed >= GAME_DURATION or self._lives <= 0:
            self._game_over = True
            self._over_time = now
            return

        self._wrists = [None, None]
        if landmarks:
            lw = landmark_to_px(landmarks[LEFT_WRIST],  frame_w, frame_h)
            rw = landmark_to_px(landmarks[RIGHT_WRIST], frame_w, frame_h)
            if self.MIRROR_HANDS:
                logical_left_wrist  = rw
                logical_right_wrist = lw
            else:
                logical_left_wrist  = lw
                logical_right_wrist = rw
            self._wrists = [logical_left_wrist, logical_right_wrist]

        if self._current_action == "rest":
            if now - self._action_start >= self._action_timeout:
                self._set_action(self._sequence_index + 1)
        else:
            if landmarks:
                self._process_punch(now)
            if now - self._action_start >= self._action_timeout:
                self._advance_action(failed=True)

        self._effects = [(x, y, bt) for x, y, bt in self._effects if now - bt < 0.4]

    # ------------------------------------------------------------------
    def _current_instruction(self) -> Tuple[str, str]:
        if self._current_action == "punch_left":
            return "GOLPE IZQUIERDO", "Golpea con la mano izquierda"
        if self._current_action == "punch_right":
            return "GOLPE DERECHO", "Golpea con la mano derecha"
        return "DESCANSO", "Recupérate y mantente listo"

    def render(self, frame: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        canvas = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (10, 12, 20)

        # Background glow strips
        cv2.rectangle(canvas, (0, 0), (self.CANVAS_W, 140), (15, 20, 34), -1)
        cv2.rectangle(canvas, (0, 560), (self.CANVAS_W, self.CANVAS_H), (10, 12, 18), -1)
        cv2.line(canvas, (40, 80), (1240, 80), (45, 100, 140), 2, cv2.LINE_AA)

        # Main panels
        draw_rounded_panel(canvas, self.GAME_AREA, color=(15, 18, 40), alpha=0.85, radius=26)
        draw_rounded_panel(canvas, (self.CAM_AREA[0] - 14, self.CAM_AREA[1] - 14,
                                    self.CAM_AREA[2] + 28, self.CAM_AREA[3] + 28),
                           color=(20, 24, 40), alpha=0.76, radius=20)
        draw_rounded_panel(canvas, (40, 18, 1200, 100), color=(18, 22, 36), alpha=0.88, radius=22)

        # HUD
        draw_text(canvas, "MOVEPLAY BOXING", (60, 52), scale=1.45, color=CYAN, thickness=3)
        draw_text(canvas, f"SCORE: {self._score}", (60, 110), scale=0.9, color=WHITE)
        draw_text(canvas, f"COMBO x{self._combo}", (320, 110), scale=0.82,
                  color=CYAN if self._combo > 1 else WHITE)
        draw_text(canvas, f"LIVES: {self._lives}", (530, 110), scale=0.82, color=RED)

        elapsed = now - self._start
        remaining = max(0.0, GAME_DURATION - elapsed)
        draw_text(canvas, f"TIME: {remaining:.0f}s", (780, 110), scale=0.9, color=WHITE)

        # Trainer sprite
        trainer_sprite = self._current_trainer_sprite()
        overlay_image_alpha(canvas, trainer_sprite, (76, 130))

        # Instructions
        title, subtitle = self._current_instruction()
        draw_text(canvas, title, (460, 210), scale=1.8, color=YELLOW, thickness=4)
        draw_text(canvas, subtitle, (460, 260), scale=0.85, color=WHITE)

        # Camera preview
        camera_preview = cv2.resize(frame, (self.CAM_AREA[2], self.CAM_AREA[3]), interpolation=cv2.INTER_AREA)
        if self._landmarks:
            try:
                from core.renderer import draw_skeleton
                draw_skeleton(camera_preview, self._landmarks, POSE_CONNECTIONS,
                              self.CAM_AREA[2], self.CAM_AREA[3], joint_color=YELLOW,
                              bone_color=CYAN, joint_radius=3, thickness=2)
            except Exception:
                pass
        canvas[self.CAM_AREA[1]:self.CAM_AREA[1] + self.CAM_AREA[3],
               self.CAM_AREA[0]:self.CAM_AREA[0] + self.CAM_AREA[2]] = camera_preview
        draw_text(canvas, "CAMERA", (self.CAM_AREA[0] + 12, self.CAM_AREA[1] + 28),
                  scale=0.7, color=WHITE)

        # Current target
        if self._current_target is not None:
            mapped_center = self._map_point(self._current_target.center)
            mapped_radius = self._map_radius(self._current_target.radius)
            draw_target(canvas, mapped_center, mapped_radius, self._current_target.color, hit=self._current_target.hit)

        # Hit effects
        for x, y, born in self._effects:
            age = now - born
            r   = int(20 + age * 140)
            alpha_val = max(0, int(255 * (1.0 - age / 0.4)))
            overlay = canvas.copy()
            mapped_center = self._map_point((x, y))
            cv2.circle(overlay, mapped_center, r, CYAN, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha_val / 255, canvas, 1 - alpha_val / 255, 0, canvas)

        for pt in self._wrists:
            if pt is not None:
                draw_wrist_cursor(canvas, self._map_point(pt), color=YELLOW)

        # Feedback box
        if self._feedback_msg and now - self._feedback_time < 1.5:
            draw_panel(canvas, (460, 300, 680, 60), color=(30, 30, 60), alpha=0.78)
            draw_text(canvas, self._feedback_msg, (480, 340), scale=1.2, color=self._feedback_color, thickness=3)

        # Bottom progress and stage info
        bar_x, bar_y = 60, self.CANVAS_H - 60
        bar_w, bar_h = 1120, 28
        draw_rounded_panel(canvas, (bar_x - 10, bar_y - 10, bar_w + 20, bar_h + 20),
                           color=(15, 18, 30), alpha=0.84, radius=18)
        bar_color = GREEN if remaining > 20 else ORANGE if remaining > 10 else RED
        draw_progress_bar(canvas, (bar_x, bar_y), (bar_w, bar_h), remaining, GAME_DURATION, fg_color=bar_color)
        draw_text(canvas, "SIGUE AL ENTRENADOR Y NO TE RELAJES", (bar_x, bar_y - 14),
                  scale=0.75, color=WHITE)

        # Game over state overlay
        if self._game_over:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.CANVAS_W, self.CANVAS_H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)
            draw_text(canvas, "GAME OVER", (self.CANVAS_W // 2 - 185, self.CANVAS_H // 2 - 30),
                      scale=2.0, color=RED, thickness=5)
            draw_text(canvas, f"Puntuacion final: {self._score}",
                      (self.CANVAS_W // 2 - 190, self.CANVAS_H // 2 + 40),
                      scale=1.0, color=YELLOW, thickness=3)

        return canvas

    @property
    def next_state(self) -> Optional[str]:
        return self._next

    @property
    def name(self) -> str:
        return "boxing"
