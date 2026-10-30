"""Corpus loading, encoding, and querying - shared by dashboard.py (terminal
UI) and mcp_server.py (MCP tools) so both operate on the exact same tested
retrieval logic instead of duplicating it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .backends.on_device import OnDeviceBackend
from .router import EmbeddingRouter

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
_VIDEO_SAMPLE_FRAMES = 3


@dataclass(frozen=True)
class DocItem:
    label: str  # shown in the results table
    kind: str  # "text" | "image" | "video"
    source: str  # literal text for "text", a filesystem path otherwise


def load_docs(path: Path) -> list[DocItem]:
    """Each line is either literal text, or a path (relative to this file)
    to an image/video to embed with the vision encoder instead."""
    if not path.exists():
        raise FileNotFoundError(f"Document corpus not found at {path} (one document per line)")
    base_dir = path.parent
    items: list[DocItem] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        candidate = base_dir / line
        ext = candidate.suffix.lower()
        if ext in _IMAGE_EXTS and candidate.is_file():
            items.append(DocItem(label=line, kind="image", source=str(candidate)))
        elif ext in _VIDEO_EXTS and candidate.is_file():
            items.append(DocItem(label=line, kind="video", source=str(candidate)))
        else:
            items.append(DocItem(label=line, kind="text", source=line))
    if not items:
        raise ValueError(f"Document corpus at {path} is empty")
    return items


def sample_video_frames(path: str, num_frames: int = _VIDEO_SAMPLE_FRAMES) -> list[bytes]:
    """Evenly-spaced frames, PNG-encoded so they go through the same
    encode_image() path as still images."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not read any frames from video {path}")
        frames: list[bytes] = []
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 1) / (num_frames + 1)))
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".png", frame)
            if ok:
                frames.append(buf.tobytes())
        if not frames:
            raise ValueError(f"Could not sample any frames from video {path}")
        return frames
    finally:
        cap.release()


def encode_docs(on_device: OnDeviceBackend, items: Sequence[DocItem]) -> np.ndarray:
    vecs: list[np.ndarray | None] = [None] * len(items)

    text_idx = [i for i, it in enumerate(items) if it.kind == "text"]
    if text_idx:
        text_vecs = on_device.encode_text([items[i].source for i in text_idx], prompt_name="Document")
        for i, v in zip(text_idx, text_vecs):
            vecs[i] = v

    image_idx = [i for i, it in enumerate(items) if it.kind == "image"]
    if image_idx:
        blobs = [Path(items[i].source).read_bytes() for i in image_idx]
        image_vecs = on_device.encode_image(blobs)
        for i, v in zip(image_idx, image_vecs):
            vecs[i] = v

    # No native video encoder on this model (text/vision/audio only, see
    # config.py) - pool a few sampled frames through the vision encoder
    # instead, then re-normalize (the mean of unit vectors isn't unit length,
    # and encode()'s cosine-via-dot-product trick elsewhere assumes it is).
    for i, it in enumerate(items):
        if it.kind != "video":
            continue
        frame_vecs = on_device.encode_image(sample_video_frames(it.source))
        pooled = frame_vecs.mean(axis=0)
        vecs[i] = pooled / np.linalg.norm(pooled)

    return np.stack(vecs)


def run_query(
    router: EmbeddingRouter, query_text: str, doc_vecs, docs: Sequence[DocItem]
) -> tuple[list[tuple[DocItem, float]], float]:
    start = time.monotonic()
    query_vec = router.encode_text([query_text], prompt_name="SearchQuery")[0]
    scores = doc_vecs @ query_vec  # both L2-normalized -> dot product == cosine sim
    ranked = sorted(zip(docs, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    return ranked, time.monotonic() - start
