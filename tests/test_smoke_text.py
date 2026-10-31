"""Reproduces the manual smoke test run during setup - but through the
actual app pipeline (OnDeviceBackend -> EmbeddingRouter -> corpus.run_query
against docs/corpus.txt), the same code path dashboard.py and
mcp_server.py use, rather than calling SentenceTransformer directly against
a hardcoded, duplicate copy of the corpus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_service import corpus  # noqa: E402
from embedding_service.backends.on_device import OnDeviceBackend  # noqa: E402
from embedding_service.router import EmbeddingRouter  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "models" / "embeddinggemma-2"
DOCS_FILE = Path(__file__).parent.parent / "docs" / "corpus.txt"


def test_text_similarity_ranks_relevant_doc_highest():
    on_device = OnDeviceBackend(MODEL_DIR, device="cpu")
    router = EmbeddingRouter(on_device)

    docs = corpus.load_docs(DOCS_FILE)
    doc_vecs = corpus.encode_docs(on_device, docs)
    assert doc_vecs.shape[1] == 768

    ranked, elapsed = corpus.run_query(router, "What causes the northern lights?", doc_vecs, docs)

    top_doc, top_score = ranked[0]
    assert "aurora" in top_doc.label.lower()
    assert top_score > 0.7
    assert top_score > ranked[1][1]
    assert elapsed >= 0


if __name__ == "__main__":
    test_text_similarity_ranks_relevant_doc_highest()
    print("ok")
