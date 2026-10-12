from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class EmbeddingBackend(ABC):
    """Common interface so callers don't care whether inference ran on-device or in the cloud."""

    @abstractmethod
    def encode_text(self, texts: Sequence[str], *, prompt_name: str | None = None) -> np.ndarray:
        ...

    def encode_image(self, images: Sequence[bytes]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} was not loaded with the vision encoder")

    def encode_audio(self, clips: Sequence[bytes]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} was not loaded with the audio encoder")

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can currently serve requests (model loaded / endpoint reachable)."""
