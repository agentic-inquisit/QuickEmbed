# Desktop / edge box (CPU) — Ollama or llama.cpp

This is the quantized, CPU-friendly equivalent of the full-precision path
documented in PROPOSAL.md — use this path for an actual edge box (NUC,
Raspberry Pi 5 class, old laptop) rather than the raw bf16 safetensors
checkpoint, which is heavier than it needs to be for CPU-only inference.

## Ollama

```bash
ollama pull embeddinggemma-2:270m-bf16-text   # text-only, smallest
ollama pull embeddinggemma-2                   # full multimodal tag

curl http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma-2",
  "input": "task: search result | query: What causes the northern lights?"
}'
```

Ollama doesn't support `prompt_name` — apply the `task: ... | query: ...` /
`title: none | text: ...` templates yourself before sending the request (see
`src/embedding_service/backends/cloud.py` for the same pattern against vLLM).

## llama.cpp (GGUF, for the smallest footprint / no daemon)

```bash
# model: ggml-org/embeddinggemma-2-GGUF (or unsloth/embeddinggemma-2-GGUF)
./llama-embedding -m embeddinggemma-2-Q4_K_M.gguf -p "your text here"
```

## Constraints
- Both paths are text-first in practice today; GGUF multimodal (vision/audio
  mmproj) support for this specific checkpoint should be verified against
  the exact llama.cpp/Ollama release you pin, since GGUF multimodal support
  lags text by model family.
- Quantization (Q4_K_M / INT4) trades a small recall hit for ~4x smaller
  weights — fine for most RAG use cases, verify on your own eval set if
  retrieval quality is critical.
