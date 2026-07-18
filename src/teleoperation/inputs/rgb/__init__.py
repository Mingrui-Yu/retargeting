"""RGB callback decoding with optional MediaPipe dependencies."""

__all__ = ["SingleHandDetector", "decode_rgb_sample"]


def __getattr__(name: str):
    """Load the optional RGB detector implementation only on demand.

    Args:
        name: Public attribute requested from the RGB sensor package.

    Returns:
        Requested RGB decoder or detector.
    """
    if name in __all__:
        from teleoperation.inputs.rgb.common import SingleHandDetector, decode_rgb_sample

        return {"SingleHandDetector": SingleHandDetector, "decode_rgb_sample": decode_rgb_sample}[name]
    raise AttributeError(name)
