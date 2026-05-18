import numpy as np
from typing import Tuple


def calc_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle in degrees at vertex b, formed by segments b→a and b→c."""
    va = np.array(a, dtype=float) - np.array(b, dtype=float)
    vc = np.array(c, dtype=float) - np.array(b, dtype=float)
    cosine = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def landmark_to_px(lm, w: int, h: int) -> Tuple[int, int]:
    return (int(lm.x * w), int(lm.y * h))
