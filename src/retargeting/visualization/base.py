from __future__ import annotations

from typing import Protocol


class ReplayVisualizer(Protocol):
    """Display backend for offline replay frames."""

    def update_frame(self, frame: object) -> None:
        ...
