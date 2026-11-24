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


if __name__ == "__main__":
    unittest.main()
