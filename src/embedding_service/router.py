"""Picks on-device vs cloud per request.

Policy is intentionally simple: prefer on-device (private, free, works offline);
fall back to cloud only when the on-device backend can't serve the request
(modality not loaded locally, or explicitly unavailable).
"""

from typing import Sequence

import numpy as np

from .backends.base import EmbeddingBackend


class EmbeddingRouter:
    def __init__(self, on_device: EmbeddingBackend, cloud: EmbeddingBackend | None = None):
        self.on_device = on_device
        self.cloud = cloud

    def encode_text(self, texts: Sequence[str], *, prompt_name: str | None = None) -> np.ndarray:
        if self.on_device.is_available:
            return self.on_device.encode_text(texts, prompt_name=prompt_name)
        if self.cloud is not None and self.cloud.is_available:
            return self.cloud.encode_text(texts, prompt_name=prompt_name)
        raise RuntimeError("no embedding backend available (on-device unavailable, cloud unset/unreachable)")

    def encode_image(self, images: Sequence[bytes]) -> np.ndarray:
        try:
            return self.on_device.encode_image(images)
        except NotImplementedError:
            if self.cloud is not None:
                return self.cloud.encode_image(images)
            raise

    def encode_audio(self, clips: Sequence[bytes]) -> np.ndarray:
        try:
            return self.on_device.encode_audio(clips)
        except NotImplementedError:
            if self.cloud is not None:
                return self.cloud.encode_audio(clips)
            raise
