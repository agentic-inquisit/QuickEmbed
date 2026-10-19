"""Single-screen terminal dashboard: type a query, see ranked results, watch
live compute usage (this process + system + optional cloud/GPU) the whole time.

Threading model (see PROPOSAL.md / the approved plan for the full rationale):
  - Thread A (main): drains submitted query lines, polls the in-flight encode
    Future. Never touches the terminal directly.
  - Thread B (rich.Live's own refresh thread): the only thing that ever draws
    to the screen; also where MetricsSampler.sample() runs each tick.
  - Thread C (this module's input reader): reads raw keystrokes in cbreak
    mode and writes them into AppState, which Thread B renders.
"""

from __future__ import annotations

import os
import queue
import sys
import termios
import threading
import time
import tty
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backends.cloud import CloudBackend
from .backends.on_device import OnDeviceBackend
from .metrics import MetricsSampler, SystemSnapshot
from .router import EmbeddingRouter

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "embeddinggemma-2"
DOCS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "corpus.txt"


def _load_docs(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Document corpus not found at {path} (one document per line)")
    docs = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not docs:
        raise ValueError(f"Document corpus at {path} is empty")
    return docs


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.input_buffer: str = ""
        self.status: str = "idle"  # idle | typing | encoding | done | error
        self.last_query: str = ""
        self.results: list[tuple[str, float]] = []
        self.error: str | None = None
        self.encode_started_at: float | None = None
        self.encode_elapsed_s: float | None = None
        self.should_quit: bool = False

    def append_char(self, ch: str) -> None:
        with self._lock:
            self.input_buffer += ch
            if self.status in ("idle", "done", "error"):
                self.status = "typing"

    def backspace(self) -> None:
        with self._lock:
            self.input_buffer = self.input_buffer[:-1]

    def pop_line(self) -> str:
        with self._lock:
            line, self.input_buffer = self.input_buffer, ""
            return line

    def peek_input_buffer(self) -> str:
        with self._lock:
            return self.input_buffer

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def request_quit(self) -> None:
        with self._lock:
            self.should_quit = True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "input_buffer": self.input_buffer,
                "status": self.status,
                "last_query": self.last_query,
                "results": list(self.results),
                "error": self.error,
                "encode_started_at": self.encode_started_at,
                "encode_elapsed_s": self.encode_elapsed_s,
                "should_quit": self.should_quit,
            }


def _input_reader_loop(state: AppState, line_queue: "queue.Queue[str]") -> None:
    while True:
        ch = sys.stdin.read(1)
        if ch == "":
            continue
        if ch == "\x03":  # defensive — ISIG normally delivers SIGINT first
            state.request_quit()
            return
        if ch in ("\r", "\n"):
            line = state.pop_line()
            if line.strip():
                line_queue.put(line.strip())
            else:
                state.update(status="idle")
            continue
        if ch in ("\x7f", "\x08"):  # Backspace / DEL
            state.backspace()
            continue
        if ch == "\x1b":  # swallow escape sequences (arrow keys etc.)
            continue
        if ch.lower() == "q" and state.peek_input_buffer() == "":
            state.request_quit()
            return
        if ch.isprintable():
            state.append_char(ch)


def _run_query(
    router: EmbeddingRouter, query_text: str, doc_vecs, docs: Sequence[str]
) -> tuple[list[tuple[str, float]], float]:
    start = time.monotonic()
    query_vec = router.encode_text([query_text], prompt_name="SearchQuery")[0]
    scores = doc_vecs @ query_vec  # both L2-normalized -> dot product == cosine sim
    ranked = sorted(zip(docs, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    return ranked, time.monotonic() - start


def build_layout() -> Layout:
    root = Layout(name="root")
    root.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    root["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    root["left"].split_column(
        Layout(name="query", size=7),
        Layout(name="results"),
    )
    root["right"].split_column(
        Layout(name="system", ratio=2),
        Layout(name="process", size=6),
        Layout(name="cloud_gpu", size=9),
    )
    return root


def _bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _render_header(on_device_ready: bool, cloud_configured: bool) -> Panel:
    text = Text()
    text.append("EmbeddingGemma 2 — Live Dashboard", style="bold cyan")
    text.append("   on-device: ")
    text.append("ready" if on_device_ready else "loading", style="green" if on_device_ready else "yellow")
    text.append("   cloud: ")
    text.append("configured" if cloud_configured else "off", style="green" if cloud_configured else "dim")
    return Panel(text, style="on grey11")


def _render_query(ui: dict) -> Panel:
    body = Text()
    status = ui["status"]
    if status == "encoding":
        elapsed = time.monotonic() - ui["encode_started_at"] if ui["encode_started_at"] else 0.0
        body.append(f"> {ui['last_query']}\n", style="dim")
        body.append(f"Encoding… {elapsed:.1f}s\n", style="bold yellow")
    else:
        body.append(f"> {ui['input_buffer']}", style="bold white")
        body.append("█", style="white blink")
        body.append("\n")
        if status == "idle":
            body.append("Type a query and press Enter\n", style="dim")
        elif status == "error":
            body.append(f"Error: {ui['error']}\n", style="bold red")
        elif status == "done":
            elapsed = ui["encode_elapsed_s"] or 0.0
            body.append(f"Last query: {ui['last_query']!r}  ({elapsed:.2f}s)\n", style="dim")
    return Panel(body, title="Query", border_style="blue")


def _render_results(ui: dict) -> Panel:
    table = Table(expand=True)
    table.add_column("Rank", width=4)
    table.add_column("Score", width=7)
    table.add_column("Document")
    for i, (doc, score) in enumerate(ui["results"]):
        style = "bold green" if i == 0 else None
        snippet = doc if len(doc) <= 70 else doc[:67] + "..."
        table.add_row(str(i + 1), f"{score:.3f}", snippet, style=style)
    return Panel(table, title="Ranked Results", border_style="blue")


def _render_system(metrics: SystemSnapshot) -> Panel:
    body = Text()
    body.append(f"CPU (overall): {metrics.cpu_percent_overall:5.1f}%  {_bar(metrics.cpu_percent_overall)}\n")
    for i, pct in enumerate(metrics.cpu_percent_per_core):
        body.append(f"  core{i:<2}: {pct:5.1f}%  {_bar(pct, 10)}\n", style="dim")
    body.append(
        f"RAM: {metrics.ram_used_gb:.2f}/{metrics.ram_total_gb:.2f} GB "
        f"({metrics.ram_percent:.0f}%)  {_bar(metrics.ram_percent)}\n"
    )
    body.append(f"Disk: R {metrics.disk_read_mb_s:6.2f} MB/s  W {metrics.disk_write_mb_s:6.2f} MB/s\n")
    body.append(f"Net:  ↑ {metrics.net_sent_mb_s:6.2f} MB/s  ↓ {metrics.net_recv_mb_s:6.2f} MB/s\n")
    return Panel(body, title="System", border_style="magenta")


def _render_process(metrics: SystemSnapshot) -> Panel:
    body = Text()
    body.append(f"RSS: {metrics.proc_rss_mb:8.1f} MB\n")
    body.append(f"CPU: {metrics.proc_cpu_percent:6.1f}%  {_bar(min(metrics.proc_cpu_percent, 100))}\n")
    return Panel(body, title="This process", border_style="magenta")


def _render_cloud_gpu(metrics: SystemSnapshot) -> Panel:
    body = Text()
    body.append(f"Local GPU: {metrics.gpu_status}\n")
    cloud_style = {"ok": "green", "unreachable": "red", "not_configured": "dim"}[metrics.cloud_status]
    body.append(f"Cloud:     {metrics.cloud_status}\n", style=cloud_style)
    if metrics.cloud_status == "ok":
        body.append(f"  requests_running = {metrics.cloud_num_requests_running}\n")
        body.append(f"  gpu_cache_usage  = {metrics.cloud_gpu_cache_usage_pct}%\n")
    return Panel(body, title="Cloud / GPU", border_style="magenta")


def _render_footer(ui: dict) -> Panel:
    text = Text("[Enter] submit   [q] quit (when input empty)   [Ctrl-C] quit", style="dim")
    return Panel(text)


def render_dashboard(
    ui: dict,
    metrics: SystemSnapshot,
    docs: Sequence[str],
    on_device_ready: bool,
    cloud_configured: bool,
) -> Layout:
    layout = build_layout()
    layout["header"].update(_render_header(on_device_ready, cloud_configured))
    layout["query"].update(_render_query(ui))
    layout["results"].update(_render_results(ui))
    layout["system"].update(_render_system(metrics))
    layout["process"].update(_render_process(metrics))
    layout["cloud_gpu"].update(_render_cloud_gpu(metrics))
    layout["footer"].update(_render_footer(ui))
    return layout


def main() -> None:
    console = Console()
    console.print("Loading EmbeddingGemma 2 (on-device)…")

    docs_path = Path(os.environ.get("EMBEDDING_DOCS_FILE", str(DOCS_FILE)))
    docs = _load_docs(docs_path)

    on_device = OnDeviceBackend(MODEL_DIR)
    cloud_url = os.environ.get("EMBEDDING_CLOUD_URL")
    cloud = CloudBackend(cloud_url) if cloud_url else None
    router = EmbeddingRouter(on_device, cloud)

    doc_vecs = on_device.encode_text(docs, prompt_name="Document")
    sampler = MetricsSampler(cloud_metrics_url=f"{cloud.base_url}/metrics" if cloud else None)

    state = AppState()
    line_queue: "queue.Queue[str]" = queue.Queue()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="encode")
    pending_future: Future | None = None

    def build_renderable():
        ui = state.snapshot()
        metrics = sampler.sample()
        return render_dashboard(ui, metrics, docs, on_device.is_available, cloud is not None)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    input_thread = threading.Thread(target=_input_reader_loop, args=(state, line_queue), daemon=True)
    input_thread.start()

    try:
        with Live(
            get_renderable=build_renderable,
            console=console,
            screen=True,
            refresh_per_second=4,
            redirect_stdout=False,
            redirect_stderr=False,
        ):
            while not state.snapshot()["should_quit"]:
                if pending_future is None:
                    try:
                        q = line_queue.get_nowait()
                    except queue.Empty:
                        q = None
                    if q is not None:
                        state.update(
                            status="encoding", last_query=q, encode_started_at=time.monotonic(), error=None
                        )
                        pending_future = executor.submit(_run_query, router, q, doc_vecs, docs)
                elif pending_future.done():
                    try:
                        results, elapsed = pending_future.result()
                        state.update(status="done", results=results, encode_elapsed_s=elapsed)
                    except Exception as exc:
                        state.update(status="error", error=str(exc))
                    pending_future = None
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
