"""Live Vision Pro acquisition using the shared AVP decoder."""

from __future__ import annotations

from typing import Any

from teleoperation.inputs.avp.common import decode_avp_sample
from teleoperation.types import SensorHandSample


class AvpOnlineInput:
    """Own the optional VisionProStreamer connection and polling lifecycle."""

    def __init__(self, avp_ip: str) -> None:
        """Configure a live AVP endpoint without importing its optional client.

        Args:
            avp_ip: Vision Pro streamer network address.

        Returns:
            None.
        """
        if not avp_ip:
            raise ValueError("avp_ip must not be empty.")
        self.avp_ip = str(avp_ip)
        self._streamer: Any | None = None
        self._source_index = 0

    def open(self) -> None:
        """Connect to VisionProStreamer through a lazy optional import.

        Args:
            None.

        Returns:
            None.
        """
        from avp_stream import VisionProStreamer

        self._streamer = VisionProStreamer(ip=self.avp_ip, record=True)
        self._source_index = 0

    def read(self) -> SensorHandSample:
        """Poll and decode the latest live AVP frame.

        Args:
            None.

        Returns:
            Decoded sample; a missing live frame has empty hand fields.
        """
        if self._streamer is None:
            raise RuntimeError("AvpOnlineInput must be opened before read().")
        sample = decode_avp_sample(self._streamer.latest, source_index=self._source_index)
        self._source_index += 1
        return sample

    def reset(self) -> None:
        """Reset live source metadata for a new flow cycle.

        Args:
            None.

        Returns:
            None.
        """
        self._source_index = 0

    def close(self) -> None:
        """Close the streamer when its optional client exposes a close method.

        Args:
            None.

        Returns:
            None.
        """
        streamer = self._streamer
        self._streamer = None
        close = getattr(streamer, "close", None)
        if callable(close):
            close()
