"""Measures QuickEmbed's OnDeviceBackend wrapper overhead against the raw
sentence-transformers model it wraps, on the same inputs/hardware/process.
Part of BENCHMARK.md's Dimension 1 (wrapper overhead) - see that file for
the full comparison against TEI/Infinity/txtai/LocalAI/sentence-transformers.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from embedding_service.backends.on_device import OnDeviceBackend  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "embeddinggemma-2"
BATCH = [
    "Charged particles from the sun interact with gases in Earth's atmosphere, causing the aurora borealis.",
    "The recipe calls for two cups of flour and a teaspoon of baking soda.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The stock market fell sharply after the interest rate announcement.",
    "Quantum entanglement links particles regardless of the distance between them.",
    "Add the onions and garlic, then simmer for twenty minutes.",
    "The novel's protagonist struggles with memory loss after the accident.",
    "Mitochondria generate most of a cell's supply of ATP.",
]
N_RUNS = 20


def _timed_runs(fn, n: int) -> list[float]:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def _report(label: str, times: list[float]) -> None:
    times_ms = sorted(t * 1000 for t in times)
    mean = statistics.mean(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    throughput = len(BATCH) / (mean / 1000)
    print(f"{label:28s} mean={mean:7.2f}ms  p50={p50:7.2f}ms  p95={p95:7.2f}ms  {throughput:7.1f} docs/s")


def main() -> None:
    print(f"Model: {MODEL_DIR}")
    print(f"Batch size: {len(BATCH)} docs, {N_RUNS} runs each\n")

    raw_model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    backend = OnDeviceBackend(MODEL_DIR, device="cpu")

    # warm up both (first call pays one-time graph/cache costs)
    raw_model.encode(BATCH, prompt_name="Document", normalize_embeddings=True)
    backend.encode_text(BATCH, prompt_name="Document")

    raw_times = _timed_runs(
        lambda: raw_model.encode(BATCH, prompt_name="Document", normalize_embeddings=True), N_RUNS
    )
    wrapped_times = _timed_runs(lambda: backend.encode_text(BATCH, prompt_name="Document"), N_RUNS)

    _report("raw SentenceTransformer", raw_times)
    _report("QuickEmbed OnDeviceBackend", wrapped_times)

    overhead_pct = (statistics.mean(wrapped_times) / statistics.mean(raw_times) - 1) * 100
    print(f"\nWrapper overhead: {overhead_pct:+.1f}%")


if __name__ == "__main__":
    main()
