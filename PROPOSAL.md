# QuickEmbed — EmbeddingGemma 2 Deployment Proposal

## What this is

Google DeepMind released **EmbeddingGemma 2** on 2026-10-06: a 740M-parameter,
Apache 2.0, open-weight embedding model built on the Gemma 4 architecture. It
maps text, code, images, video, and audio into a single 768-dimensional
space, with Matryoshka Representation Learning (MRL) letting you truncate
vectors to 512/256/128 dims, and an 8,192-token context window. It's modular:
you only load the encoders you need.

| Variant | Params | Encoders loaded |
|---|---|---|
| Text + code only | 270M (130M transformer + 140M embedder) | text |
| + vision | 440M | text, vision (170M) |
| + audio | 570M | text, audio (300M) |
| Full multimodal | 740M | text, vision, audio |

No hosted API exists yet (Gemini Enterprise Model Garden support is
"coming soon" per Google) — as of today, every deployment is self-hosted.

## System constraints

- **RAM (on-device, quantized INT4/INT8 QAT)**: ~191MB active for text-only,
  ~567MB for full multimodal, measured by Google on a Pixel 11 Pro.
- **RAM (unquantized, as downloaded)**: the full bf16 checkpoint is 1.5GB on
  disk (`model.safetensors`); loads to ~3GB resident on CPU (transformers
  upcasts bf16 math on CPU).
- **GPU VRAM (cloud serving)**: not 16GB — that was headroom, not a floor.
  Weights are ~1.5GB; embedding inference has no KV-cache growth like text
  generation does. 4–6GB VRAM comfortably serves single-request traffic;
  more only matters for large concurrent batch sizes.
- **No safety tuning** — Google ships this without a safety-tuning pass.
  Don't feed it untrusted user content without your own input sanitization
  if the embeddings feed anything security-sensitive (e.g. access control).
- **Uneven language coverage** across the 100 languages it supports — verify
  retrieval quality on your own eval set if non-English is a priority.
- **vLLM's OpenAI-compatible embeddings endpoint is text-only** for this
  model as of this release. Vision/audio must be served on-device or via a
  dedicated multimodal serving path (e.g. TEI) — not through the vLLM path
  used here.
- **vLLM doesn't support `prompt_name`/automatic prompt templating** for
  embeddings — the `"task: search result | query: {}"` /
  `"title: none | text: {}"` templates must be applied client-side before
  the request goes out (see `backends/cloud.py`).

## Pipeline

```
                    ┌─────────────────────┐
   query/doc text ─▶│   EmbeddingRouter    │
                    │  (prefers on-device, │
                    │   falls back cloud)  │
                    └──────────┬───────────┘
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       OnDeviceBackend              CloudBackend (Phase 2)
   (sentence-transformers,       (OpenAI-compatible /v1/embeddings,
    local safetensors,            vLLM on a single GPU — see
    CPU or edge accelerator)       deploy/cloud/)
                 │                           │
                 └─────────────┬─────────────┘
                                ▼
                     768-dim vector (or MRL-
                     truncated to 512/256/128)
                                │
                                ▼
                        VectorStore (Qdrant)
```

Both backends implement the same `EmbeddingBackend` interface
(`backends/base.py`), so callers never need to know which one answered a
given request — only the router does.

## Deployment plan (phased, as directed)

### Phase 1 — on-device first (primary target: mobile / browser / edge)

The actual edge targets (phone, browser, embedded box) load Google's
pre-converted bundles, not the raw safetensors checkpoint:

| Target | Artifact | Notes |
|---|---|---|
| Android/iOS | `litert-community/embeddinggemma-2-*-litert-lm` (MediaPipe `.task`) | pick 270M/440M/740M bundle by modality need — see `deploy/ondevice/mobile/` |
| Browser | `onnx-community/embeddinggemma-2-ONNX` via `onnxruntime-web`/WebGPU, WASM fallback | text-first; route image/audio to cloud — see `deploy/ondevice/browser/` |
| Desktop/edge box (CPU) | GGUF via llama.cpp, or Ollama (`ollama pull embeddinggemma-2`) | quantized, no GPU needed — see `deploy/ondevice/ollama/` |

Local serving in this repo uses the full unquantized `google/embeddinggemma-2`
checkpoint, downloaded to `models/embeddinggemma-2/` and served through
`OnDeviceBackend` (`sentence-transformers`, CPU).

### Phase 2 — cloud inference option (for when on-device isn't enough)

`deploy/cloud/docker-compose.vllm.yml` brings up a single-GPU vLLM server
exposing an OpenAI-compatible `/v1/embeddings` endpoint plus a Prometheus
`/metrics` endpoint. `CloudBackend` talks to it; `EmbeddingRouter` only
falls back to it when on-device can't serve the request. Use this tier for:
devices too constrained to run even the 270M text bundle, battery-saving
mode, or throughput beyond what on-device inference gives you.

A natural next step: a tiny resource policy in the router
(RAM/battery/connectivity-aware) to decide *when* to prefer cloud over
on-device, rather than only falling back on hard failure as it does today.
