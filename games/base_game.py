from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseGame(ABC):
    @abstractmethod
    def update(self, frame: np.ndarray, landmarks: Optional[list],
               frame_w: int, frame_h: int) -> None:
        """Process input and advance game state (called every frame)."""

    @abstractmethod
    def render(self, frame: np.ndarray) -> None:
        """Draw game elements onto frame in-place."""

    @abstractmethod
    def reset(self) -> None:
        """Restore initial state so the game can be replayed."""

    @property
    @abstractmethod
    def next_state(self) -> Optional[str]:
        """Name of the state to transition to, or None to stay."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this game."""
