"""Cloud backend: OpenAI-compatible /v1/embeddings endpoint served by vLLM or TEI on a GPU box.

This is the Phase-2 option — used when a device can't run the full 740M
multimodal model locally (low RAM, battery-saving mode, no accelerator) or
when you need higher throughput than on-device inference can give you.
"""

from typing import Sequence

import numpy as np
import requests

from ..config import NATIVE_EMBEDDING_DIM

# vLLM/TEI don't support sentence-transformers' prompt_name kwarg — the prompt
# template has to be applied client-side before the request goes out.
_PROMPT_TEMPLATES = {
    "SearchQuery": "task: search result | query: {}",
    "Document": "title: none | text: {}",
}


class CloudBackend:
    def __init__(self, base_url: str, model_name: str = "google/embeddinggemma-2", timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s

    @property
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def encode_text(self, texts: Sequence[str], *, prompt_name: str | None = None) -> np.ndarray:
        template = _PROMPT_TEMPLATES.get(prompt_name, "{}")
        formatted = [template.format(t) for t in texts]
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model_name, "input": formatted},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        return vecs

    def encode_image(self, images: Sequence[bytes]) -> np.ndarray:
        raise NotImplementedError(
            "vLLM's OpenAI-compatible embeddings endpoint is text-only as of this model's "
            "release — serve vision/audio requests on-device or add a dedicated multimodal "
            "serving path (e.g. TEI's image input) before relying on this."
        )

    def encode_audio(self, clips: Sequence[bytes]) -> np.ndarray:
        raise NotImplementedError("same caveat as encode_image — not available over this endpoint")
