"""Resident local HTTP daemon for Scuttle (`scuttle-serve`).

Keeps the 0.93 GiB packed 4-bit Scuttle model warm across Hermes / agent sessions
and automatically offloads weights back to 0.0 GiB VRAM after `--idle-offload` seconds.

Endpoints:
  GET  /health   — daemon status, VRAM usage, idle timer, cached sources
  GET  /sources  — named graphs in --graphs dir + cached dataswamps/DBs
  POST /index    — index or refresh a folder, dataswamp, SQLite DB, or JSONL/CSV
  POST /search   — run Scuttle parallel swarm search (auto-indexes & caches sources!)
  POST /score    — single-pass scoring of one page + up to 100 candidate links
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from scuttle.model import DEFAULT_MODEL_ID, ScuttleWorker, frontier_value
from scuttle.searcher import CorpusBM25, ScuttleSearcher
from scuttle.sources import load_or_build_source
from scuttle.world import World

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "scuttle"


class ScuttleDaemon:
    def __init__(
        self,
        model: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        quant: str = "4bit",
        graphs_dir: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        idle_offload: int = 120,
        max_cached_worlds: int = 8,
    ):
        self.model = model
        self.device = device
        self.quant = quant
        self.graphs_dir = Path(graphs_dir).expanduser().resolve() if graphs_dir else None
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.idle_offload = max(0, int(idle_offload))
        self.max_cached_worlds = max_cached_worlds

        self._worker: ScuttleWorker | None = None
        self._worlds: OrderedDict[str, tuple[World, CorpusBM25, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()
        self._last_used = time.monotonic()
        self._started_at = time.time()

        if self.idle_offload > 0:
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

    def _ensure_worker(self) -> ScuttleWorker:
        if self._worker is None:
            self._worker = ScuttleWorker(model_id_or_path=self.model, device=self.device, quant=self.quant)
        self._last_used = time.monotonic()
        return self._worker

    def _get_source(
        self,
        source_spec: str,
        refresh: bool = False,
        tables: list[str] | None = None,
        max_files: int = 10000,
        bridge_k: int = 6,
    ) -> tuple[World, CorpusBM25, dict[str, Any]]:
        world, info = load_or_build_source(
            source_spec,
            cache_dir=self.cache_dir / "graphs",
            graphs_dir=self.graphs_dir,
            refresh=refresh,
            tables=tables,
            max_files=max_files,
            bridge_k=bridge_k,
        )
        cache_key = info.get("cache_file") or info.get("source") or source_spec
        if not refresh and cache_key in self._worlds:
            self._worlds.move_to_end(cache_key)
            w, bm, prev_info = self._worlds[cache_key]
            return w, bm, {**prev_info, "cached": True}

        bm = CorpusBM25(world)
        self._worlds[cache_key] = (world, bm, info)
        self._worlds.move_to_end(cache_key)
        while len(self._worlds) > self.max_cached_worlds:
            self._worlds.popitem(last=False)
        return world, bm, info

    def health(self) -> dict[str, Any]:
        vram = round(torch.cuda.memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else 0.0
        idle_for = round(time.monotonic() - self._last_used, 1) if self._worker is not None else None
        return dict(
            ok=True,
            model=self.model,
            quant=self.quant,
            device=self.device,
            loaded=(self._worker is not None),
            vram_allocated_gib=vram,
            idle_offload_sec=self.idle_offload,
            idle_for_sec=idle_for,
            uptime_sec=round(time.time() - self._started_at, 1),
            cached_sources=len(self._worlds),
            pid=os.getpid(),
        )

    def list_sources(self) -> dict[str, Any]:
        named: dict[str, str] = {}
        if self.graphs_dir and self.graphs_dir.exists():
            for p in sorted(self.graphs_dir.glob("*.json*")):
                name = p.name.replace("-graph.json.gz", "").replace(".json.gz", "").replace(".json", "")
                named[name] = str(p)
        cached = [
            dict(source=info.get("source"), type=info.get("source_type"), documents=info.get("documents"), edges=info.get("edges"))
            for _, _, info in self._worlds.values()
        ]
        return dict(named_graphs=named, loaded_in_memory=cached, cache_dir=str(self.cache_dir / "graphs"))

    def index_source(self, body: dict[str, Any]) -> dict[str, Any]:
        src = body.get("source")
        if not src:
            raise ValueError("Missing required field 'source'")
        with self._lock:
            _, _, info = self._get_source(
                src,
                refresh=bool(body.get("refresh", True)),
                tables=body.get("tables"),
                max_files=int(body.get("max_files", 10000)),
                bridge_k=int(body.get("bridge_k", 6)),
            )
            # Optionally copy to a named graph in graphs_dir
            save_name = body.get("save_as")
            if save_name and self.graphs_dir and info.get("cache_file"):
                self.graphs_dir.mkdir(parents=True, exist_ok=True)
                dst = self.graphs_dir / f"{save_name}-graph.json.gz"
                import shutil
                shutil.copy2(info["cache_file"], dst)
                info["saved_named_graph"] = str(dst)
            return info

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        query = (body.get("query") or "").strip()
        src = (body.get("source") or "").strip()
        if not query:
            raise ValueError("Missing required field 'query'")
        if not src:
            raise ValueError("Missing required field 'source'")

        mode = str(body.get("mode", "scuttle")).lower()
        n_seeds = int(body.get("seeds", 4))
        budget = int(body.get("budget", 16))
        batch_k = int(body.get("batch_k", 2))
        top_k = int(body.get("top_k", 6))
        threshold = float(body.get("threshold", 0.45))
        excerpt_chars = int(body.get("excerpt_chars", 1200))
        start_ids = body.get("start_ids")
        refresh = bool(body.get("refresh", False))
        tables = body.get("tables")
        max_files = int(body.get("max_files", 10000))
        bridge_k = int(body.get("bridge_k", 6))

        t0 = time.perf_counter()
        with self._lock:
            world, bm25, info = self._get_source(
                src, refresh=refresh, tables=tables, max_files=max_files, bridge_k=bridge_k
            )

            if mode == "bm25":
                hits = bm25.search(query, k=min(top_k, budget))
                delivered = []
                for rank, did in enumerate(hits, 1):
                    p = world.papers[did]
                    delivered.append(dict(
                        doc_id=did,
                        title=p.get("title", did),
                        p_contributes=None,
                        bm25_rank=rank,
                        path=p.get("path"),
                        kind=p.get("kind"),
                        from_seed=True,
                        read_step=rank,
                        excerpt=(p.get("abstract") or "")[:excerpt_chars],
                        trail=[dict(doc_id=did, title=p.get("title", did))],
                    ))
                return dict(
                    query=query,
                    mode="bm25",
                    domain=world.domain,
                    reads_used=len(delivered),
                    budget=budget,
                    delivered=delivered,
                    frontier=[],
                    source_info=info,
                    search_ms=round(1000 * (time.perf_counter() - t0), 1),
                )

            worker = self._ensure_worker()
            searcher = ScuttleSearcher(world, worker=worker)
            searcher._bm25 = bm25
            t_s = time.perf_counter()
            res = searcher.search(
                query,
                start_ids=start_ids,
                n_seeds=n_seeds,
                budget=budget,
                batch_k=batch_k,
                contrib_threshold=threshold,
            )
            search_ms = round(1000 * (time.perf_counter() - t_s), 1)
            self._last_used = time.monotonic()

        # Enrich delivered items with exact path/kind/excerpt for Hermes
        chosen = res["contributing"][:top_k] if res["contributing"] else res["top_pages"][:top_k]
        delivered = []
        for item in chosen:
            p = world.papers.get(item["doc_id"], {})
            delivered.append(dict(
                doc_id=item["doc_id"],
                title=item["title"],
                p_contributes=item["p_contributes"],
                path=p.get("path"),
                kind=p.get("kind"),
                table=p.get("table"),
                from_seed=item["from_seed"],
                read_step=item["read_step"],
                excerpt=(p.get("abstract") or "")[:excerpt_chars],
                trail=item["trail"],
            ))

        frontier = []
        for f in res.get("top_frontier", [])[:12]:
            p = world.papers.get(f["doc_id"], {})
            frontier.append(dict(
                doc_id=f["doc_id"],
                title=f["title"],
                priority=f["priority"],
                path=p.get("path"),
                kind=p.get("kind"),
            ))

        return dict(
            query=query,
            mode="scuttle",
            model=self.model,
            quant=self.quant,
            domain=world.domain,
            reads_used=res["reads_used"],
            budget=budget,
            n_contributing=len(res["contributing"]),
            delivered=delivered,
            frontier=frontier,
            source_info=info,
            search_ms=search_ms,
            total_ms=round(1000 * (time.perf_counter() - t0), 1),
        )

    def score(self, body: dict[str, Any]) -> dict[str, Any]:
        query = (body.get("query") or "").strip()
        cur = body.get("current") or {}
        cands = body.get("candidates") or []
        if not query or not cur:
            raise ValueError("Missing required fields 'query' and 'current'")

        cur_id = str(cur.get("doc_id") or cur.get("page_id") or "current")
        papers = {cur_id: dict(title=str(cur.get("title") or ""), abstract=str(cur.get("text") or cur.get("abstract") or ""))}
        out_edges = []
        cand_ids = []
        for i, c in enumerate(cands[:100]):
            cid = str(c.get("doc_id") or c.get("page_id") or f"cand_{i}")
            if cid == cur_id or cid in papers:
                cid = f"{cid}_{i}"
            papers[cid] = dict(title=str(c.get("title") or cid), abstract=str(c.get("abstract") or c.get("description") or ""))
            out_edges.append([cid, str(c.get("context") or "")])
            cand_ids.append(cid)

        tmp_world = World(dict(domain="adhoc", doc_noun="document", papers=papers, out={cur_id: out_edges}), max_links=100)
        t0 = time.perf_counter()
        with self._lock:
            worker = self._ensure_worker()
            row = worker.menu_row(tmp_world, query, cur_id)
            pz, lz = worker.score_rows([row])[0]
            p_page = float(torch.sigmoid(pz).item())
            l_probs = torch.sigmoid(lz)
            vals = frontier_value(l_probs).cpu().tolist() if len(row["dests"]) else []
            p_d0 = l_probs[:, 0].cpu().tolist() if len(row["dests"]) else []
            self._last_used = time.monotonic()

        scored = sorted(
            (
                dict(
                    doc_id=did,
                    title=papers[did]["title"],
                    priority=round(v, 4),
                    p_direct_hit=round(pd0, 4),
                )
                for did, v, pd0 in zip(row["dests"], vals, p_d0)
            ),
            key=lambda x: -x["priority"],
        )
        return dict(
            query=query,
            doc_id=cur_id,
            p_contributes=round(p_page, 4),
            candidates=scored,
            score_ms=round(1000 * (time.perf_counter() - t0), 1),
        )


def make_handler(daemon: ScuttleDaemon):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep daemon logs quiet & crisp

        def _send(self, code: int, payload: dict[str, Any]):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8")) if n > 0 else {}

        def do_GET(self):
            try:
                if self.path == "/health":
                    self._send(200, daemon.health())
                elif self.path == "/sources":
                    self._send(200, daemon.list_sources())
                else:
                    self._send(404, {"error": f"Unknown GET endpoint: {self.path}"})
            except Exception as e:
                self._send(500, {"error": str(e)})

        def do_POST(self):
            try:
                body = self._read_json()
                if self.path == "/search":
                    self._send(200, daemon.search(body))
                elif self.path == "/index":
                    self._send(200, daemon.index_source(body))
                elif self.path == "/score":
                    self._send(200, daemon.score(body))
                else:
                    self._send(404, {"error": f"Unknown POST endpoint: {self.path}"})
            except FileNotFoundError as e:
                self._send(404, {"error": str(e)})
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                self._send(500, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(prog="scuttle-serve", description="Resident Scuttle HTTP daemon for Hermes & AI agents")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quant", choices=["4bit", "bf16", "none", "bnb4", "8bit"], default="4bit")
    ap.add_argument("--graphs", default=os.environ.get("SCUTTLE_GRAPHS"), help="Optional directory of prebuilt .json.gz graphs")
    ap.add_argument("--cache-dir", default=os.environ.get("SCUTTLE_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    ap.add_argument("--idle-offload", type=int, default=120, help="Seconds idle before unloading VRAM (0=keep warm)")
    a = ap.parse_args()

    daemon = ScuttleDaemon(
        model=a.model,
        device=a.device,
        quant=a.quant,
        graphs_dir=a.graphs,
        cache_dir=a.cache_dir,
        idle_offload=a.idle_offload,
    )
    pid_file = Path(a.cache_dir).expanduser().resolve() / "serve.pid"
    pid_file.write_text(str(os.getpid()))

    server = ThreadingHTTPServer((a.host, a.port), make_handler(daemon))
    print(f"scuttle-serve listening on http://{a.host}:{a.port} (model={a.model}, quant={a.quant}, idle_offload={a.idle_offload}s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
