"""Pure-logic tests for src/embedding_service/corpus.py - no real model
needed, uses a fake backend, so these run without the 1.5GB checkpoint
(and without torch/transformers even importing - see corpus.py's
TYPE_CHECKING-gated OnDeviceBackend import).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_service import corpus  # noqa: E402
from embedding_service.backends.base import EmbeddingBackend  # noqa: E402
from embedding_service.router import EmbeddingRouter  # noqa: E402


class FakeOnDevice(EmbeddingBackend):
    """Deterministic stand-in for OnDeviceBackend - a real subclass of the
    EmbeddingBackend ABC (not just duck-typed), so it's held to the same
    contract OnDeviceBackend/CloudBackend are, and can be wrapped in the
    real EmbeddingRouter instead of needing a separate fake router."""

    def __init__(self, text_vec: np.ndarray | None = None, expected_prompt: str | None = None):
        self._text_vec = text_vec if text_vec is not None else np.array([1.0, 0.0, 0.0])
        self._expected_prompt = expected_prompt

    @property
    def is_available(self) -> bool:
        return True

    def encode_text(self, texts, *, prompt_name=None):
        if self._expected_prompt is not None:
            assert prompt_name == self._expected_prompt
        return np.stack([self._text_vec for _ in texts])

    def encode_image(self, images):
        v = np.array([0.0, 1.0, 1.0])
        v = v / np.linalg.norm(v)
        return np.stack([v for _ in images])


@pytest.fixture
def mixed_corpus(tmp_path: Path) -> Path:
    import cv2

    img_path = tmp_path / "pic.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype="uint8"))

    vid_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (10, 10))
    for i in range(10):
        writer.write(np.full((10, 10, 3), i * 20, dtype="uint8"))
    writer.release()

    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("a plain text doc\npic.png\nclip.mp4\nanother text doc\n")
    return corpus_file


def test_load_docs_dispatches_by_file_type(mixed_corpus: Path):
    items = corpus.load_docs(mixed_corpus)
    assert [it.kind for it in items] == ["text", "image", "video", "text"]
    assert items[0].source == "a plain text doc"
    assert items[1].source == str(mixed_corpus.parent / "pic.png")
    assert items[2].source == str(mixed_corpus.parent / "clip.mp4")


def test_load_docs_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        corpus.load_docs(tmp_path / "does_not_exist.txt")


def test_load_docs_empty_file_raises(tmp_path: Path):
    empty = tmp_path / "corpus.txt"
    empty.write_text("\n\n")
    with pytest.raises(ValueError):
        corpus.load_docs(empty)


def test_sample_video_frames_returns_requested_count(mixed_corpus: Path):
    vid_path = mixed_corpus.parent / "clip.mp4"
    frames = corpus.sample_video_frames(str(vid_path))
    assert len(frames) == corpus._VIDEO_SAMPLE_FRAMES
    assert all(isinstance(f, bytes) and len(f) > 0 for f in frames)


def test_encode_docs_routes_by_kind_and_preserves_order(mixed_corpus: Path):
    items = corpus.load_docs(mixed_corpus)
    vecs = corpus.encode_docs(FakeOnDevice(), items)

    assert vecs.shape == (4, 3)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "every vector must be unit-norm"

    # text docs (indices 0, 3) got the text encoder's output
    assert np.allclose(vecs[0], [1.0, 0.0, 0.0])
    assert np.allclose(vecs[3], [1.0, 0.0, 0.0])
    # image doc (index 1) and video doc (index 2, pooled from frames of the
    # same fake image encoder) both land on the vision encoder's output
    assert np.allclose(vecs[1], vecs[2])
    assert not np.allclose(vecs[1], vecs[0])


def test_run_query_ranks_by_cosine_similarity_descending():
    # 4 docs with known analytic similarity to the query, by construction:
    # identical / 45deg / orthogonal / opposite.
    query_dir = np.array([1.0, 0.0, 0.0])
    doc_vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.70710678, 0.70710678, 0.0],
        ]
    )
    docs = [corpus.DocItem(label=f"doc{i}", kind="text", source=f"doc{i}") for i in range(4)]

    router = EmbeddingRouter(FakeOnDevice(text_vec=query_dir, expected_prompt="SearchQuery"))
    ranked, elapsed = corpus.run_query(router, "irrelevant", doc_vecs, docs)

    labels = [d.label for d, _ in ranked]
    scores = [s for _, s in ranked]

    assert labels == ["doc0", "doc3", "doc1", "doc2"]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.70710678)
    assert scores[2] == pytest.approx(0.0)
    assert scores[3] == pytest.approx(-1.0)
    assert elapsed >= 0
