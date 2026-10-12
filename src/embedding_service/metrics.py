"""Point-in-time system/process/GPU/cloud metric sampling.

Only ever call MetricsSampler.sample() from a single thread (in dashboard.py
that's Live's own refresh thread) — the delta-based counters below assume a
single caller and keep no locks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil
import requests


@dataclass
class SystemSnapshot:
    cpu_percent_overall: float
    cpu_percent_per_core: list[float]
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    proc_rss_mb: float
    proc_cpu_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    net_sent_mb_s: float
    net_recv_mb_s: float
    gpu_status: str
    cloud_status: str  # "not_configured" | "unreachable" | "ok"
    cloud_num_requests_running: float | None
    cloud_gpu_cache_usage_pct: float | None


class MetricsSampler:
    def __init__(
        self,
        cloud_metrics_url: str | None = None,
        cloud_timeout_s: float = 1.5,
        cloud_poll_interval_s: float = 2.0,
    ):
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(None)  # prime: first real call needs a baseline
        psutil.cpu_percent(percpu=False)  # prime overall-CPU baseline
        psutil.cpu_percent(percpu=True)  # prime per-core baseline (separate internal state)
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_time = time.monotonic()

        self._cloud_metrics_url = cloud_metrics_url
        self._cloud_timeout_s = cloud_timeout_s
        self._cloud_poll_interval_s = cloud_poll_interval_s
        self._cloud_last_fetch = 0.0
        self._cloud_cache: tuple[str, float | None, float | None] = (
            "not_configured" if cloud_metrics_url is None else "unreachable",
            None,
            None,
        )

        self._gpu_handle = None
        self._gpu_init_attempted = False

    def sample(self) -> SystemSnapshot:
        now = time.monotonic()
        dt = max(now - self._prev_time, 1e-6)

        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        disk_read_mb_s = (disk.read_bytes - self._prev_disk.read_bytes) / dt / 1e6
        disk_write_mb_s = (disk.write_bytes - self._prev_disk.write_bytes) / dt / 1e6
        net_sent_mb_s = (net.bytes_sent - self._prev_net.bytes_sent) / dt / 1e6
        net_recv_mb_s = (net.bytes_recv - self._prev_net.bytes_recv) / dt / 1e6
        self._prev_disk, self._prev_net, self._prev_time = disk, net, now

        cpu_overall = psutil.cpu_percent(percpu=False)  # non-blocking; delta since last call
        cpu_per_core = psutil.cpu_percent(percpu=True)
        vm = psutil.virtual_memory()
        proc_rss_mb = self._process.memory_info().rss / 1e6
        proc_cpu_percent = self._process.cpu_percent(None)  # non-blocking; delta since last call

        gpu_status = self._sample_gpu()
        cloud_status, cloud_running, cloud_cache_pct = self._sample_cloud(now)

        return SystemSnapshot(
            cpu_percent_overall=cpu_overall,
            cpu_percent_per_core=cpu_per_core,
            ram_used_gb=vm.used / 1e9,
            ram_total_gb=vm.total / 1e9,
            ram_percent=vm.percent,
            proc_rss_mb=proc_rss_mb,
            proc_cpu_percent=proc_cpu_percent,
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            net_sent_mb_s=net_sent_mb_s,
            net_recv_mb_s=net_recv_mb_s,
            gpu_status=gpu_status,
            cloud_status=cloud_status,
            cloud_num_requests_running=cloud_running,
            cloud_gpu_cache_usage_pct=cloud_cache_pct,
        )

    def _sample_gpu(self) -> str:
        # Import deferred to call time (not module scope) so metrics.py stays
        # importable on GPU-less hosts even if pynvml is absent.
        try:
            import pynvml
        except ImportError:
            return "No local GPU (pynvml not installed)"
        try:
            if not self._gpu_init_attempted:
                self._gpu_init_attempted = True
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if self._gpu_handle is None:
                return "No local GPU"
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            return f"{util.gpu}% util, {mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB"
        except Exception:
            # NVML missing driver / no device / permission error — degrade, don't crash.
            self._gpu_handle = None
            return "No local GPU"

    def _sample_cloud(self, now: float) -> tuple[str, float | None, float | None]:
        if self._cloud_metrics_url is None:
            return "not_configured", None, None
        if now - self._cloud_last_fetch < self._cloud_poll_interval_s:
            return self._cloud_cache  # reuse last result between polls
        self._cloud_last_fetch = now
        try:
            resp = requests.get(self._cloud_metrics_url, timeout=self._cloud_timeout_s)
            resp.raise_for_status()
            # vLLM's own queue-depth/KV-cache-usage proxy for GPU load, not raw
            # nvidia-smi utilization/VRAM — zero extra infra needed on the GPU box.
            running = _parse_prom_metric(resp.text, "vllm:num_requests_running")
            cache_pct = _parse_prom_metric(resp.text, "vllm:gpu_cache_usage_perc")
            self._cloud_cache = ("ok", running, cache_pct)
        except requests.RequestException:
            self._cloud_cache = ("unreachable", None, None)
        return self._cloud_cache


def _parse_prom_metric(text: str, metric_name: str) -> float | None:
    """Minimal Prometheus text-exposition-format line matcher (no prometheus_client dep).

    Matches a line whose metric name (the part before an optional `{labels}`
    block) equals `metric_name` exactly, and returns the trailing numeric value.
    """
    for line in text.splitlines():
        if not line or line.startswith("#") or not line.startswith(metric_name):
            continue
        rest = line[len(metric_name):]
        if rest and rest[0] not in (" ", "{", "\t"):
            continue  # e.g. metric_name_total sharing our prefix — not a real match
        try:
            return float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return None
