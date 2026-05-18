import os
import cv2
import numpy as np
from typing import Tuple, List, Optional

# BGR color palette
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (0,   0,   255)
GREEN  = (0,   255, 0)
BLUE   = (255, 0,   0)
YELLOW = (0,   255, 255)
CYAN   = (255, 255, 0)
ORANGE = (0,   165, 255)
PURPLE = (255, 0,   255)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list,
    connections: List[Tuple[int, int]],
    frame_w: int,
    frame_h: int,
    joint_color: Tuple = GREEN,
    bone_color: Tuple = WHITE,
    joint_radius: int = 5,
    thickness: int = 2,
) -> None:
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    vis = [
        (getattr(lm, 'visibility', None) or 1.0) > 0.3
        for lm in landmarks
    ]
    for a, b in connections:
        if vis[a] and vis[b]:
            cv2.line(frame, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        if vis[i]:
            cv2.circle(frame, (px, py), joint_radius, joint_color, -1, cv2.LINE_AA)


def draw_text(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.7,
    color: Tuple = WHITE,
    thickness: int = 2,
) -> None:
    """Draw text with a black outline for readability on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    draw_text(frame, f"FPS: {fps:.0f}", (10, 28), scale=0.6, color=YELLOW, thickness=1)


def draw_panel(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple = (15, 15, 50),
    alpha: float = 0.72,
) -> None:
    """Semi-transparent filled rectangle."""
    x, y, w, h = rect
    x1 = max(0, x);  y1 = max(0, y)
    x2 = min(frame.shape[1], x + w);  y2 = min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bg  = np.full_like(roi, color)
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1, cv2.LINE_AA)


def draw_rounded_panel(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple = (15, 15, 50),
    alpha: float = 0.72,
    radius: int = 22,
) -> None:
    """Semi-transparent rounded panel with subtle border."""
    x, y, w, h = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 2, cv2.LINE_AA)


def load_sprite(
    path: str,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load a PNG sprite with alpha channel, or return a placeholder when missing."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        placeholder = np.zeros((size[1] if size else 420, size[0] if size else 420, 4), np.uint8)
        placeholder[:] = (46, 52, 69, 255)
        label = os.path.splitext(os.path.basename(path))[0].upper()
        cv2.putText(placeholder, label, (16, placeholder.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA)
        return placeholder
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.dstack([img, alpha])
    if size is not None and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def overlay_image_alpha(
    frame: np.ndarray,
    overlay: np.ndarray,
    pos: Tuple[int, int],
) -> None:
    """Blend an RGBA overlay image onto a BGR frame using alpha channel."""
    x, y = pos
    h, w = overlay.shape[:2]
    if overlay.ndim != 3 or overlay.shape[2] != 4:
        return
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    overlay_x1 = x1 - x
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y1 = y1 - y
    overlay_y2 = overlay_y1 + (y2 - y1)
    overlay_slice = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    alpha = overlay_slice[..., 3:] / 255.0
    blended = overlay_slice[..., :3].astype(float) * alpha + frame[y1:y2, x1:x2].astype(float) * (1.0 - alpha)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)


def draw_progress_bar(
    frame: np.ndarray,
    pos: Tuple[int, int],
    size: Tuple[int, int],
    value: float,
    max_value: float,
    bg_color: Tuple = (50, 50, 50),
    fg_color: Tuple = GREEN,
) -> None:
    x, y = pos
    w, h = size
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(w * min(max(value / (max_value or 1), 0.0), 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1)


def draw_button(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    hover_ratio: float = 0.0,
    base_color: Tuple = (30, 30, 110),
    hover_color: Tuple = (60, 60, 200),
) -> None:
    x, y, w, h = rect
    color = hover_color if hover_ratio > 0 else base_color
    draw_panel(frame, rect, color=color, alpha=0.75)

    if hover_ratio > 0:
        cx, cy = x + w // 2, y + h // 2
        r = min(w, h) // 2 - 4
        end_angle = int(360 * hover_ratio)
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, end_angle, CYAN, 3, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    draw_text(frame, label, (tx, ty), scale=0.85, color=WHITE, thickness=2)


def draw_target(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple,
    hit: bool = False,
) -> None:
    if hit:
        cv2.circle(frame, center, radius + 12, YELLOW, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, WHITE, 2, cv2.LINE_AA)
    cv2.circle(frame, center, max(4, radius // 2), WHITE, 2, cv2.LINE_AA)


def draw_wrist_cursor(frame: np.ndarray, pos: Optional[Tuple[int, int]], color: Tuple = CYAN) -> None:
    if pos is None:
        return
    cv2.circle(frame, pos, 14, color, 2, cv2.LINE_AA)
    cv2.circle(frame, pos, 3, WHITE, -1, cv2.LINE_AA)
