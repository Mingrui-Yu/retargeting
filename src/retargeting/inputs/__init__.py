"""Input data contracts and device adapters for canonical hand observations.

Detector implementations are intentionally not imported here, so importing the
retargeting core does not require optional RGB/MediaPipe dependencies.
"""

from retargeting.inputs.base import HandInput, HandObservation

__all__ = ["HandInput", "HandObservation"]
