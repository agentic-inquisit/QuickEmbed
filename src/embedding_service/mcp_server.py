"""MCP server exposing QuickEmbed's retrieval as tools, so an MCP client
(Claude, or anything else) can call into it directly instead of only a
human at dashboard.py's terminal UI.

Run directly: PYTHONPATH=src python3 -m embedding_service.mcp_server
See README.md for client registration (e.g. a project's .mcp.json).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer

from . import corpus
from .backends.cloud import CloudBackend
from .backends.on_device import OnDeviceBackend
from .router import EmbeddingRouter

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "embeddinggemma-2"
DOCS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "corpus.txt"


@dataclass
class AppContext:
    router: EmbeddingRouter
    on_device: OnDeviceBackend
    docs: list[corpus.DocItem]
    doc_vecs: object  # np.ndarray; untyped here to avoid importing numpy just for an annotation
    docs_path: Path


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[AppContext]:
    # Same setup as dashboard.main(), minus the Rich/terminal/threading
    # pieces - done once at server startup, not per request, since encoding
    # is the slow part (~2.7s/8-doc batch on CPU, measured in BENCHMARK.md).
    docs_path = Path(os.environ.get("EMBEDDING_DOCS_FILE", str(DOCS_FILE)))
    docs = corpus.load_docs(docs_path)

    needed_modalities = {"text"}
    if any(item.kind in ("image", "video") for item in docs):
        needed_modalities.add("vision")

    on_device = OnDeviceBackend(MODEL_DIR, modalities=frozenset(needed_modalities))
    cloud_url = os.environ.get("EMBEDDING_CLOUD_URL")
    cloud = CloudBackend(cloud_url) if cloud_url else None
    router = EmbeddingRouter(on_device, cloud)

    doc_vecs = await anyio.to_thread.run_sync(corpus.encode_docs, on_device, docs)

    yield AppContext(router=router, on_device=on_device, docs=docs, doc_vecs=doc_vecs, docs_path=docs_path)


mcp = MCPServer("quickembed", lifespan=lifespan)


@mcp.tool()
async def search_corpus(query: str, ctx: Context, top_k: int = 5) -> list[dict]:
    """Rank the loaded corpus (text/image/video) by cosine similarity to `query`."""
    app: AppContext = ctx.request_context.lifespan_context
    ranked, _elapsed = await anyio.to_thread.run_sync(
        corpus.run_query, app.router, query, app.doc_vecs, app.docs
    )
    return [{"label": d.label, "kind": d.kind, "score": score} for d, score in ranked[:top_k]]


@mcp.tool()
async def embed_text(texts: list[str], ctx: Context) -> list[list[float]]:
    """Return raw L2-normalized text embeddings for `texts` (not scored against the corpus)."""
    app: AppContext = ctx.request_context.lifespan_context
    vecs = await anyio.to_thread.run_sync(app.router.encode_text, texts)
    return vecs.tolist()


@mcp.tool()
def corpus_info(ctx: Context) -> dict:
    """Report what's currently loaded: doc count, modalities, model, corpus file."""
    app: AppContext = ctx.request_context.lifespan_context
    return {
        "doc_count": len(app.docs),
        "modalities": sorted(app.on_device.modalities),
        "model_dir": str(MODEL_DIR),
        "docs_path": str(app.docs_path),
    }


@mcp.tool()
async def reload_corpus(ctx: Context) -> dict:
    """Re-read the corpus file and re-embed it in place, without restarting
    the server (which would re-pay the model load cost)."""
    app: AppContext = ctx.request_context.lifespan_context
    app.docs = corpus.load_docs(app.docs_path)
    app.doc_vecs = await anyio.to_thread.run_sync(corpus.encode_docs, app.on_device, app.docs)
    return {"doc_count": len(app.docs)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
