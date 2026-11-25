"""Unit tests for the standalone Scuttle package (World, Abel summation, Markdown indexer, BM25, MCP)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scuttle.indexer import index_directory
from scuttle.model import ABEL_W, THRESHOLDS, frontier_value
from scuttle.quant import PackedInt4Embedding, PackedInt4Linear
from scuttle.searcher import CorpusBM25
from scuttle.world import World, load_world


class TestScuttle(unittest.TestCase):
    def test_abel_summation(self):
        expected = {0: 1.00, 1: 0.50, 2: 0.28, 3: 0.16, 4: 0.09, 5: 0.04, 6: 0.04, None: 0.00}
        for d, exp in expected.items():
            p = torch.tensor([1.0 if (d is not None and d <= t) else 0.0 for t in THRESHOLDS])
            self.assertAlmostEqual(float(frontier_value(p, ABEL_W)), exp, places=6)

    def test_flex_100_menu(self):
        # Node 0 has 10 outgoing and 95 incoming -> flex menu should fill 10 + 90 = 100!
        papers = {str(i): dict(title=f"Doc {i}", abstract=f"Body {i}.") for i in range(110)}
        out = {"0": [[str(i), f"links {i}"] for i in range(1, 11)]}
        for i in range(11, 106):
            out[str(i)] = [["0", f"{i} cites 0"]]
        w = World(dict(papers=papers, out=out), max_links=100, min_out=60)
        m0 = w.menu("0")
        self.assertEqual(len(m0), 100)
        self.assertEqual(sum(1 for _, k, _ in m0 if k == "cites"), 10)
        self.assertEqual(sum(1 for _, k, _ in m0 if k == "cited by"), 90)

    def test_markdown_indexer_and_bm25(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vault"
            root.mkdir()
            (root / "index.md").write_text("# Start\nSee [[Regulation D]] and [Exemptions](exemptions.md) for details.")
            (root / "Regulation D.md").write_text("# Regulation D\nCovers private offerings. Also check [[Accredited Investor]].")
            (root / "exemptions.md").write_text("# Securities Exemptions\nPrivate placement rules reference [[Regulation D]].")
            (root / "Accredited Investor.md").write_text("# Accredited Investor\nDefines net worth and income criteria for private offerings.")

            out_p = Path(td) / "vault-graph.json.gz"
            info = index_directory(root, out_p, domain="notes", doc_noun="note")
            self.assertEqual(info["documents"], 4)
            self.assertGreaterEqual(info["edges"], 4)

            w = load_world(out_p)
            bm = CorpusBM25(w)
            hits = bm.search("net worth income accredited investor", k=2)
            self.assertEqual(hits[0], "Accredited Investor.md")

    def test_mcp_stdio_handshake_and_tools_list(self):
        reqs = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        ]
        proc = subprocess.run(
            [sys.executable, "-m", "scuttle.mcp_server"],
            input="\n".join(reqs) + "\n",
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["result"]["serverInfo"]["name"], "scuttle-mcp")
        tool_names = [t["name"] for t in lines[1]["result"]["tools"]]
        self.assertIn("scuttle_search", tool_names)
        self.assertIn("scuttle_index_directory", tool_names)

    def test_packed_int4_linear_and_embedding(self):
        torch.manual_seed(0)
        lin = torch.nn.Linear(128, 64, bias=False)
        qlin = PackedInt4Linear.from_linear(lin, group_size=64)
        self.assertEqual(qlin.qdata.dtype, torch.uint8)
        self.assertEqual(qlin.qdata.numel(), 128 * 64 // 2)
        x = torch.randn(3, 128)
        y_ref = lin(x)
        y_q = qlin(x)
        cos = torch.nn.functional.cosine_similarity(y_ref.flatten(), y_q.flatten(), dim=0).item()
        self.assertGreater(cos, 0.995)

        emb = torch.nn.Embedding(32, 128)
        qemb = PackedInt4Embedding.from_embedding(emb, group_size=64)
        self.assertEqual(qemb.qdata.shape, (32, 64))
        ids = torch.tensor([[0, 5, 17]])
        e_ref = emb(ids)
        e_q = qemb(ids)
        cos_e = torch.nn.functional.cosine_similarity(e_ref.flatten(), e_q.flatten(), dim=0).item()
        self.assertGreater(cos_e, 0.995)

    def test_dataswamp_and_sqlite_indexers(self):
        import sqlite3
        from scuttle.sources import index_dataswamp, index_sqlite, load_or_build_source

        with tempfile.TemporaryDirectory() as td:
            # 1. Mixed dataswamp: markdown with frontmatter + python code import + image stub + unlinked txt
            swamp = Path(td) / "swamp"
            swamp.mkdir()
            (swamp / "notes.md").write_text(
                "---\ntitle: Auth Overview\naliases:\n  - AuthDoc\n---\n"
                "Covers JWT rotation. See [[diagram.png]] and [Tokens](tokens.txt).\n"
            )
            (swamp / "tokens.txt").write_text("Ed25519 signing keys rotate via JWKS dual-key window.")
            (swamp / "orphan.txt").write_text("JWKS dual-key rotation prevents dropped Ed25519 sessions.")
            (swamp / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (swamp / "auth.py").write_text("import utils\ndef verify_jwt(): return True\n")
            (swamp / "utils.py").write_text("def parse_jwks(): return {}\n")

            graph, info = index_dataswamp(swamp, bridge_k=4)
            self.assertEqual(info["documents"], 6)
            self.assertGreaterEqual(info["explicit_edges"], 3)  # wikilink img + mdlink txt + py import
            self.assertEqual(graph["papers"]["diagram.png"]["kind"], "image")

            # 2. Read-only SQLite database with explicit foreign key + heuristic *_id
            db_path = Path(td) / "vault.sqlite"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, summary TEXT);
                CREATE TABLE notes (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id),
                    title TEXT,
                    body TEXT
                );
                INSERT INTO projects VALUES (1, 'Apollo', 'Distributed consensus engine written in Rust');
                INSERT INTO notes VALUES (10, 1, 'Raft Election Bug', 'Fixed split-vote timeout on leader election');
                INSERT INTO notes VALUES (11, 1, 'WAL Compaction', 'Snapshotting cuts recovery latency by 4x');
                """
            )
            conn.commit()
            conn.close()

            g_db, db_info = index_sqlite(db_path, bridge_k=2)
            self.assertEqual(db_info["documents"], 3)
            self.assertGreaterEqual(db_info["explicit_edges"], 2)  # notes:10 -> projects:1, notes:11 -> projects:1
            self.assertIn("projects:1", [dst for dst, _ in g_db["out"]["notes:10"]])

            # 3. Cache hit check via load_or_build_source
            cache_dir = Path(td) / "cache"
            w1, i1 = load_or_build_source(str(db_path), cache_dir=cache_dir)
            self.assertFalse(i1["cached"])
            w2, i2 = load_or_build_source(str(db_path), cache_dir=cache_dir)
            self.assertTrue(i2["cached"])
            self.assertEqual(len(w1.papers), len(w2.papers))


if __name__ == "__main__":
    unittest.main()
