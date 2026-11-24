"""Zero-bloat JSON-RPC 2.0 stdio MCP server for Scuttle (`scuttle-mcp`).

Defaults to pure-PyTorch packed 4-bit + 4-bit embeddings (0.93 GiB static VRAM, ~100% BF16 accuracy,
zero bitsandbytes dependency!) and supports `--idle-offload` (evicts weights after N idle seconds
so idle VRAM = 0.0 GiB).

Example MCP config (Antigravity / Claude Code / Cursor / Codex):
{
  "mcpServers": {
    "scuttle": {
      "command": "scuttle-mcp",
      "args": ["--idle-offload", "120", "--graphs", "/path/to/graphs"]
    }
  }
}
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

from scuttle.indexer import index_directory
from scuttle.model import DEFAULT_MODEL_ID, ScuttleWorker
from scuttle.searcher import CorpusBM25, ScuttleSearcher
from scuttle.world import load_world

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "scuttle_search",
        "description": (
            "Search a linked document graph (citations, legal statutes, wikis, markdown vaults) using Scuttle's "
            "1.7B frontier-calibrated parallel swarm. Returns ranked contributing documents with calibrated "
            "P(contributes) scores and exact link provenance trails in ~1 second."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "graph": {"type": "string", "description": "Registered graph name or path to graph.json(.gz)"},
                "query": {"type": "string", "description": "Natural-language question or concept to search for"},
                "seeds": {"type": "integer", "default": 4, "description": "BM25 auto-seeds (ignored if start_ids set)"},
                "budget": {"type": "integer", "default": 16, "description": "Total document reads across swarm"},
                "start_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit start doc IDs (zero-index crawl)"},
                "threshold": {"type": "number", "default": 0.45, "description": "Min P(contributes) to surface"},
            },
            "required": ["graph", "query"],
        },
    },
    {
        "name": "scuttle_index_directory",
        "description": "Index a local directory of Markdown ([[wikilinks]] / [links](file.md)), HTML, or text files into a Scuttle graph.json.gz.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory of linked files to index"},
                "output_path": {"type": "string", "description": "Destination path for graph.json.gz"},
                "domain": {"type": "string", "default": "docs"},
                "doc_noun": {"type": "string", "default": "document"},
            },
            "required": ["directory", "output_path"],
        },
    },
    {
        "name": "scuttle_list_graphs",
        "description": "List pre-registered Scuttle graphs available in the server's graph directory.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class ScuttleEngine:
    def __init__(self, model=DEFAULT_MODEL_ID, device="auto", quant="4bit", graph_dir=None, idle_offload=0):
        self.model = model
        self.device = device
        self.quant = quant
        self.graph_dir = Path(graph_dir) if graph_dir else None
        self.idle_offload = idle_offload
        self._worker = None
        self._worlds = {}
        self._bm25s = {}
        self._lock = threading.Lock()
        self._last_used = time.monotonic()

        if idle_offload > 0:
            t = threading.Thread(target=self._reaper, daemon=True)
            t.start()

    def _reaper(self):
        while True:
            time.sleep(10)
            with self._lock:
                if self._worker is not None and (time.monotonic() - self._last_used) >= self.idle_offload:
                    self._worker = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _ensure_worker(self):
        if self._worker is None:
            self._worker = ScuttleWorker(model_id_or_path=self.model, device=self.device, quant=self.quant)
        self._last_used = time.monotonic()
        return self._worker

    def list_graphs(self):
        out = {}
        if self.graph_dir and self.graph_dir.exists():
            for p in sorted(self.graph_dir.glob("*.json*")):
                name = p.name.replace("-graph.json.gz", "").replace(".json.gz", "").replace(".json", "")
                out[name] = str(p)
        return out

    def _resolve_graph(self, graph_arg):
        avail = self.list_graphs()
        if graph_arg in avail:
            return avail[graph_arg]
        return graph_arg

    def _get_world(self, path):
        if path not in self._worlds:
            w = load_world(path, max_links=100)
            self._worlds[path] = w
            self._bm25s[path] = CorpusBM25(w)
        return self._worlds[path], self._bm25s[path]

    def call_tool(self, name, args):
        if name == "scuttle_list_graphs":
            return self.list_graphs()
        if name == "scuttle_index_directory":
            info = index_directory(
                args["directory"],
                args["output_path"],
                domain=args.get("domain", "docs"),
                doc_noun=args.get("doc_noun", "document"),
            )
            return info
        if name == "scuttle_search":
            with self._lock:
                worker = self._ensure_worker()
                gpath = self._resolve_graph(args["graph"])
                world, bm25 = self._get_world(gpath)
                searcher = ScuttleSearcher(world, worker=worker)
                searcher._bm25 = bm25
                t0 = time.perf_counter()
                res = searcher.search(
                    args["query"],
                    start_ids=args.get("start_ids"),
                    n_seeds=int(args.get("seeds", 4)),
                    budget=int(args.get("budget", 16)),
                    contrib_threshold=float(args.get("threshold", 0.45)),
                )
                res["search_ms"] = round(1000 * (time.perf_counter() - t0), 1)
                self._last_used = time.monotonic()
                return res
        raise ValueError(f"Unknown tool: {name}")


def _send(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(prog="scuttle-mcp", description="Scuttle stdio MCP server for AI agents")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quant", choices=["4bit", "bf16", "none", "bnb4", "8bit"], default="4bit",
                    help="4bit (default: 0.93 GiB pure-PyTorch packed) | bf16 | bnb4 | 8bit")
    ap.add_argument("--graphs", default=os.environ.get("SCUTTLE_GRAPHS"), help="Optional directory of .json.gz graphs")
    ap.add_argument("--idle-offload", type=int, default=0, help="Seconds idle before unloading model from VRAM (0=keep warm)")
    a = ap.parse_args()

    engine = ScuttleEngine(model=a.model, device=a.device, quant=a.quant, graph_dir=a.graphs, idle_offload=a.idle_offload)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "scuttle-mcp", "version": "0.1.0"},
                },
            })
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            try:
                out = engine.call_tool(params.get("name"), params.get("arguments") or {})
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"content": [{"type": "text", "text": json.dumps(out, indent=2, ensure_ascii=False)}]},
                })
            except Exception as e:
                _send({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Error: {e}"}]},
                })
        elif rid is not None:
            _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
