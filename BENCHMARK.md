# QuickEmbed vs. the field — benchmark plan + scan results

## 1. What QuickEmbed actually is (repo scan)

| | |
|---|---|
| Size | 883 LOC across 10 Python files (`src/embedding_service/`) |
| Core | Thin wrapper around `sentence-transformers` (`OnDeviceBackend`) + an OpenAI-compatible-client `CloudBackend` (vLLM), picked by `EmbeddingRouter` |
| Interface | **Terminal dashboard only** (`dashboard.py`, Rich-based) — no HTTP/gRPC API, no CLI flags, no SDK |
| Modalities | Text (wired), image (wired, vision encoder), video (frame-sampled + pooled through vision encoder — no native video encoder on the model) |
| Vector store | Qdrant via a 39-line wrapper (`vector_store.py`), MRL-truncated to 256d — **not wired into the dashboard**, only used ad hoc |
| Tests | One smoke test (`tests/test_smoke_text.py`) that needs the real 1.5GB model downloaded; no CI config found |
| Packaging | None — no `pyproject.toml`/`setup.py`, no `LICENSE` file, install is `pip install -r requirements.txt` |
| Deployment docs | `deploy/ondevice/{mobile,browser,ollama}/README.md` + `deploy/cloud/docker-compose.vllm.yml` — **documented, not exercised** (confirmed in README: "not exercised in this dev container") |
| Git history | 3 commits, single contributor, no remote activity beyond the initial push |

**Honest framing**: QuickEmbed is a personal demo/reference implementation for one specific model (EmbeddingGemma 2), not a general-purpose serving engine. That's the right comparison frame — it competes on *clarity and multimodal demo value*, not on throughput, API surface, or production features.

## 2. The 5 comparison projects (selected + verified via live search, 2026-10-19)

| Project | Stars | License | Scope |
|---|---|---|---|
| [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | 19.2k | Apache-2.0 | The library QuickEmbed directly wraps — baseline for wrapper overhead |
| [neuml/txtai](https://github.com/neuml/txtai) | ~13.0k | Apache-2.0 | Embeddings database: vector search + SQL + graph + multimodal indexing + pipelines, closest in *spirit* to QuickEmbed's "search over docs" demo |
| [mudler/LocalAI](https://github.com/mudler/LocalAI) | 49.6k | MIT | Broadest self-hosted multimodal API (LLM/vision/audio/image/video + embeddings), OpenAI-compatible, Docker-first |
| [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) (TEI) | ~5.1k | Apache-2.0/HFOIL | Production-grade Rust embedding server, dynamic batching, OpenAI-compatible — the throughput benchmark to beat |
| [michaelfeil/infinity](https://github.com/michaelfeil/infinity) | ~3.0k | MIT | Closest architectural peer: multimodal (text/CLIP/CLAP/ColPali) serving engine, OpenAI-compatible, multi-backend (PyTorch/ONNX/TensorRT) |

Rationale for this set: it spans the whole stack QuickEmbed touches — the library it's built on (sentence-transformers), a production single-purpose server to benchmark throughput against (TEI), the closest multimodal-serving peer (Infinity), a broader multimodal platform (LocalAI), and a search/demo-app peer (txtai) — rather than 5 arbitrary "popular" repos.

## 3. Benchmark dimensions + methodology

| Dimension | Method | Status |
|---|---|---|
| **Wrapper overhead** | Time `OnDeviceBackend.encode_text()` vs. raw `SentenceTransformer.encode()` on the same model/inputs/hardware | ✅ **Run** — `benchmarks/bench_overhead.py`, real `google/embeddinggemma-2` weights, this sandbox (CPU-only, 8-doc batch, 20 runs): raw mean 2847ms (2.8 docs/s) vs. wrapped mean 2687ms (3.0 docs/s) → **-5.6% "overhead"**, i.e. statistically no overhead — `OnDeviceBackend.encode_text` is a direct pass-through to `model.encode()`, confirmed empirically, not just by reading the code. Also confirms PROPOSAL.md's own caveat: ~3 docs/s unquantized bf16-on-CPU is slow enough that the GPU/cloud tier (or a quantized on-device bundle) matters for anything beyond this single-user terminal demo. |
| **Feature coverage** | Matrix: HTTP API, batching, auth, multi-model serving, reranking, vector store integration, quantization support | ✅ Done below, from scanned code + verified docs |
| **Modality coverage** | Matrix: text / image / video / audio, native vs. improvised (e.g. QuickEmbed's frame-pooling for video) | ✅ Done below |
| **Deployment surface** | Docker image? Mobile/edge/browser targets? GPU required? | ✅ Done below |
| **Throughput/latency vs. TEI/Infinity** | Same model, same hardware, HTTP load test (e.g. `hey`/`wrk` against each server's `/embed` endpoint) | ⏳ Phase 2 — needs installing Rust TEI binary + Infinity's PyTorch stack side by side; out of scope for this pass (hours of setup, not minutes) |
| **Maturity/community** | Stars, contributors, CI presence, release cadence | ✅ Done above + below |

## 4. Feature + modality matrix

| | QuickEmbed | sentence-transformers | TEI | Infinity | txtai | LocalAI |
|---|---|---|---|---|---|---|
| HTTP API | ❌ (terminal only) | ❌ (library) | ✅ OpenAI-compatible | ✅ OpenAI-compatible | ✅ FastAPI | ✅ OpenAI-compatible |
| Text embeddings | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Image embeddings | ✅ (vision encoder) | ✅ (model-dependent) | ❌ (text-only) | ✅ (CLIP) | ✅ | ✅ |
| Video embeddings | ✅ (frame-pool, improvised) | ❌ | ❌ | ❌ | ✅ (documented) | ✅ |
| Audio embeddings | 🟡 (stubbed, not implemented) | ✅ (model-dependent) | ❌ | ✅ (CLAP) | ✅ | ✅ |
| Vector store built in | 🟡 (Qdrant wrapper, not wired to the demo) | ❌ | ❌ | ❌ | ✅ (native) | 🟡 (via backends) |
| Multi-model serving | ❌ (one model, hardcoded path) | N/A (library) | ❌ (one model per instance) | ✅ | ✅ | ✅ |
| Dynamic batching | ❌ | N/A | ✅ | ✅ | 🟡 | 🟡 |
| Live resource dashboard | ✅ (unique to QuickEmbed) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Tests / CI | 🟡 (1 smoke test, no CI) | ✅ extensive | ✅ extensive | ✅ | ✅ | ✅ |
| License file present | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Packaging (pip-installable) | ❌ | ✅ | N/A (binary/Docker) | ✅ | ✅ | N/A (Docker) |

**QuickEmbed's one genuinely unique feature**: the live terminal dashboard correlating query latency with system/process/GPU/cloud resource usage in real time. None of the 5 comparables do this — they're all serving engines or libraries, not interactive instruments.

**Where it's clearly behind**: no HTTP API (can't be used as a service by anything else), no license file, no CI, no packaging, single demo-sized test suite, and the Qdrant vector store is wired but unused by the one entry point that exists.

## 5. Phase 2 — direct throughput benchmark vs. TEI / Infinity (not yet run)

Out of scope for this pass (each is a half-day+ of setup, not a follow-up to a quick scan), but concretely plannable:

1. **TEI**: `docker run ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id google/embeddinggemma-2` (confirm model support first — TEI is text-only for this model per PROPOSAL.md's own caveat, so this is a text-only comparison).
2. **Infinity**: `pip install infinity-emb[all]` or its Docker image, `infinity_emb v2 --model-id google/embeddinggemma-2` — closest apples-to-apples since it also supports the vision path.
3. Load-test both with identical batches via `hey`/`wrk`/a small Python client against `/embed` (TEI) and `/v1/embeddings` (Infinity), on this same CPU-only sandbox for a fair comparison against the Dimension-1 numbers above.
4. Record p50/p95 latency and docs/sec at batch sizes 1/8/32, same as `benchmarks/bench_overhead.py` uses, so all three numbers are directly comparable.

## 6. Bottom line

QuickEmbed is not competitive as a *serving engine* — it was never built to be one (no API, no CI, no packaging). Its differentiated value is the live dashboard correlating query latency with system/GPU/cloud load, which the field's dedicated servers don't offer. The highest-leverage next steps, in order: (1) add a `LICENSE` file — currently blocking any real-world reuse, (2) wire the existing `VectorStore` into the dashboard instead of leaving it unused, (3) add a thin FastAPI wrapper so it's usable as more than a terminal demo, (4) add CI running the smoke test against a cached model fixture.
