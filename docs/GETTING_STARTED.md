# Getting started

A step-by-step first-time setup, from a clean checkout to a working query.
For the condensed version, see [README.md](../README.md#run-it-locally);
this is the expanded walkthrough with expected output at each step.

## 1. Check prerequisites

- Python 3.10+ (`python3 --version`)
- ~2GB free disk, ~2GB free RAM
- Internet access for step 3 (first run only — the model is cached locally after)

## 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs `sentence-transformers`, `torch`, `opencv-python-headless`
(video frame sampling), `mcp` (the MCP server), and the rest of
[`requirements.txt`](../requirements.txt). On a clean CPU-only machine this
takes a few minutes — most of the time is `torch`'s wheel download.

## 3. Download the model

```bash
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('google/embeddinggemma-2', local_dir='models/embeddinggemma-2')"
```

## 4. Verify the install

```bash
PYTHONPATH=src pytest tests/test_corpus.py -v     # pure logic, no model needed, fast
PYTHONPATH=src pytest tests/test_smoke_text.py -v  # loads the real model, confirms inference works
```
## 5. Pick an entry point

QuickEmbed has two ways to actually query the corpus — pick whichever
fits what you're doing:

### Option A: the terminal dashboard (a human, interactively)

```bash
PYTHONPATH=src python3 -m embedding_service.dashboard
```

Type a query, press Enter, watch it rank `docs/corpus.txt` against your
query while showing live CPU/RAM/disk/net usage. Press `q` (with an empty
input line) or Ctrl-C to quit.

### Option B: the MCP server (an agent/tool, programmatically)

```bash
PYTHONPATH=src python3 -m embedding_service.mcp_server
```

Exposes `search_corpus`, `embed_text`, `corpus_info`, and `reload_corpus`
as MCP tools over stdio. Register it with an MCP client (e.g. a project's
`.mcp.json`):

```json
{
  "mcpServers": {
    "quickembed": {
      "command": "python3",
      "args": ["-m", "embedding_service.mcp_server"],
      "env": { "PYTHONPATH": "src" }
    }
  }
}
```

The model loads once at startup for either option — the first query/tool
call is the slow one (model load), everything after is fast relative to that.

## 6. Customize what it searches

Edit [`docs/corpus.txt`](corpus.txt) — one document per line. A line can
also be a path (relative to that file) to an image or video instead of
plain text; see the README's note on modality dispatch for how those are
embedded. Point either entry point at a different file with
`EMBEDDING_DOCS_FILE=/path/to/corpus.txt`. If the MCP server is already
running, call its `reload_corpus` tool instead of restarting it.

## Troubleshooting

See [README.md's Troubleshooting section](../README.md#troubleshooting)
for known `ModuleNotFoundError` fixes. If something else goes wrong, the
most common cause is step 3 being skipped — most errors that aren't an
import error are `OnDeviceBackend` failing to find
`models/embeddinggemma-2/`.

## Next steps

- Optional cloud/GPU backend: see the README's "Optional: cloud backend" section.
- Mobile/browser/edge deployment targets (documented, not exercised here): [`deploy/ondevice/`](../deploy/ondevice/).
- Full architecture and system constraints: [`PROPOSAL.md`](../PROPOSAL.md).