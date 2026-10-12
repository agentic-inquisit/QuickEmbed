"""On-device backend: local weights, no network call per request.

This is the Phase-1 target (laptop / edge box standing in for mobile & browser
runtimes, which load the same checkpoint via LiteRT/MLX/ONNX instead — see
deploy/ondevice/). Only the modality encoders you ask for are loaded, which
is what keeps RAM down on constrained hardware.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import MRL_TRUNCATION_DIMS, NATIVE_EMBEDDING_DIM


class OnDeviceBackend:
    def __init__(
        self,
        model_dir: str | Path,
        modalities: frozenset[str] = frozenset({"text"}),
        truncate_dim: int = NATIVE_EMBEDDING_DIM,
        device: str = "cpu",
    ):
        if truncate_dim not in MRL_TRUNCATION_DIMS:
            raise ValueError(f"truncate_dim must be one of {MRL_TRUNCATION_DIMS}")
        self.modalities = modalities
        self._model = SentenceTransformer(
            str(model_dir),
            device=device,
            truncate_dim=truncate_dim,
        )

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def encode_text(self, texts: Sequence[str], *, prompt_name: str | None = None) -> np.ndarray:
        return self._model.encode(list(texts), prompt_name=prompt_name, normalize_embeddings=True)

    def encode_image(self, images: Sequence[bytes]) -> np.ndarray:
        if "vision" not in self.modalities:
            raise NotImplementedError("vision encoder not loaded for this instance")
        from io import BytesIO
        from PIL import Image

        pil_images = [Image.open(BytesIO(b)).convert("RGB") for b in images]
        return self._model.encode(pil_images, normalize_embeddings=True)

    def encode_audio(self, clips: Sequence[bytes]) -> np.ndarray:
        if "audio" not in self.modalities:
            raise NotImplementedError("audio encoder not loaded for this instance")
        # sentence-transformers' audio extra expects raw waveform arrays or file paths;
        # wire this up against the installed sentence-transformers[audio] version before
        # relying on it — not exercised in this session (no audio sample on hand).
        raise NotImplementedError("audio path not yet wired — see note above")
