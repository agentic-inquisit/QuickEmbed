"""Static model facts for EmbeddingGemma 2 (google/embeddinggemma-2).

Values are taken from the model card, not guessed — keep this file in sync
if Google revises the checkpoint.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityEncoder:
    name: str
    params_m: int


TEXT_ENCODER = ModalityEncoder("text_code", 270)  # 130M transformer + 140M embedder
VISION_ENCODER = ModalityEncoder("vision", 170)
AUDIO_ENCODER = ModalityEncoder("audio", 300)

# Valid modular load combinations -> total params (M)
MODEL_VARIANTS = {
    frozenset({"text"}): 270,
    frozenset({"text", "vision"}): 440,
    frozenset({"text", "audio"}): 570,
    frozenset({"text", "vision", "audio"}): 740,
}

CONTEXT_WINDOW_TOKENS = 8192
NATIVE_EMBEDDING_DIM = 768
# Matryoshka Representation Learning: safe truncation targets, largest first.
MRL_TRUNCATION_DIMS = (768, 512, 256, 128)

HF_REPO_ID = "google/embeddinggemma-2"
LICENSE = "Apache-2.0"

# Quantized on-device footprint, measured by Google on Pixel 11 Pro (INT4/INT8 QAT).
ON_DEVICE_RAM_MB = {
    frozenset({"text"}): 191,
    frozenset({"text", "vision", "audio"}): 567,
}
