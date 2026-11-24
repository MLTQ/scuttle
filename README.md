# Scuttle

**Frontier-calibrated 1.7B decision model & MCP server for fast parallel swarm search over linked document graphs.**

> *Like crawling, but faster.*

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Multiqueue%2Fscuttle--1.7b-yellow)](https://huggingface.co/Multiqueue/scuttle-1.7b)
[![Demo Graphs](https://img.shields.io/badge/Demo%20Graphs-MLTQ%2Fscuttle--demos-blue)](https://github.com/MLTQ/scuttle-demos)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Weights License: LFM Open v1.0](https://img.shields.io/badge/Weights%20License-LFM%20Open%20v1.0-orange)](https://huggingface.co/Multiqueue/scuttle-1.7b/blob/main/LICENSE)

**Scuttle** gives AI agents fast, calibrated "search intuition" over linked corpora—scientific citation graphs, statutory legal codes, wikis, and Markdown/Obsidian vaults. Instead of running slow autoregressive LLM loops on every hop, Scuttle runs a **single backbone forward pass** (`LFM3-1.7B` + 15 KB of fp32 decision heads, zero text generation) to simultaneously output:

1. **$P(\text{page contributes} \mid q, \text{page})$** — a calibrated relevance probability (`0.949` AUC).
2. **$P(d \le k \mid q, \text{link})$ for $k \in \{0,1,2,3,4,6\}$** across up to **100 links at once** — calibrated frontier priorities for a shared best-first parallel swarm.

On an RTX 4090 in its default **pure-PyTorch 4-bit packed + 4-bit embedding** mode (zero `bitsandbytes` dependency!), Scuttle downloads just **973 MB**, uses **0.93 GiB static VRAM** (**1.70 GiB peak**, **0.0 GiB idle** with `--idle-offload`), retains **~100% of BF16 search accuracy**, and finishes a full multi-hop swarm search in **~1.4 seconds**.

---

## Highlights & Benchmarks

![Scuttle Benchmarks](assets/scuttle_benchmarks.png)

### 1. Search CTF Speed Run (60 Zero-Index Flags, ≤16 Reads, Warm RTX 4090)

![Scuttle CTF Speed Run](assets/scuttle_ctf_speedrun.png)

| Method | Size / VRAM | Flags Captured | Solve Rate | Latency / Run | **Seconds / Captured Flag** | LS / Law / Wiki |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Frontier Model** *(blind multi-turn cloud LLM)* | Cloud frontier | 58 / 60 | 96.7% | 64.2 s *(36.5 s med)* | 66.5 s / flag | 19/20 · 19/20 · 20/20 |
| **Scuttle 4-bit Packed (`default`)** | **1.7B / 0.93 GiB** | **46 / 60** | **76.7%** | **1.42 s** | **1.86 s / flag** | **16/20 · 17/20 · 13/20** |
| **Scuttle BF16** | 1.7B / 3.22 GiB | 45 / 60 | 75.0% | **1.19 s** | **1.59 s / flag** | 15/20 · 17/20 · 13/20 |
| `Qwen3-Reranker-0.6B` *(24-link neural)* | 0.6B / ~1.2 GiB | 22 / 60 | 36.7% | 0.65 s | 1.78 s / flag | 5/20 · 11/20 · 6/20 |
| `LFM3-1.7B` Generative Agent *(same 1.7B weights)* | 1.7B / 3.22 GiB | 10 / 60 | 16.7% | 8.43 s | 50.56 s / flag | 1/20 · 8/20 · 1/20 |
| BM25 Graph *(CPU lexical)* | 0B / CPU | 25 / 60 | 41.7% | 0.02 s | 0.04 s / flag | 8/20 · 13/20 · 4/20 |

- **vs. Cloud Frontier Model:** At **0.93 GiB local VRAM and \$0 API cost**, Scuttle captures **~80% of a frontier model's solve rate** (**85% vs. 95% on US Federal Law**, **80% vs. 95% on LitSearch**, and **`0.9575` vs. `0.9894` Page AUC** on 100 O55 test items) while running **45× faster at the mean / 26× faster at the median** (**35.7× faster per captured flag**) and **35× faster on page scoring** (`114 ms` vs. `4,003 ms`). Scuttle is built to act as the 1.4-second local scout that feeds frontier agents over MCP instead of burning 40–60s of frontier context crawling links manually.
- **vs. Same 1.7B Model as a Generative Agent:** **7.1× faster/run**, **4.5× more captures**, **31.8× faster per captured flag**.
- **vs. `Qwen3-Reranker-0.6B`:** **2.1× more captures** while scoring 4× wider link menus (100 vs. 24 links).

### 2. Controlled With vs. Without Scuttle (1,049 Queries, Exact Same 16-Doc Budget)

| Domain | Queries | Recall@16: BM25 → **Scuttle** | Top-3: BM25 → **Scuttle** | **BM25 Miss Recovery** |
| --- | ---: | :---: | :---: | ---: |
| **LitSearch** (CS/NLP citations) | 199 | 59.8% → **64.3%** (+4.5) | 37.2% → **53.3%** (+16.1) | **26 / 80 (32.5%)** |
| **eCFR** (US Federal Law) | 402 | 55.2% → **63.4%** (+8.2, $p{=}0.0008$) | 35.3% → **38.8%** (+3.5) | **63 / 180 (35.0%)** |
| **Wikispeedia** (Masked topics) | 448 | 37.5% → **61.2%** (+23.7, $p{<}10^{-20}$) | 24.6% → **52.2%** (+27.7) | **121 / 280 (43.2%)** |
| **Combined** | **1,049** | **48.5% → 62.6% (+14.1 pts)** | **31.1% → 47.3% (+16.2 pts)** | **210 / 540 (38.9%)** |

### 3. Low-VRAM Quantization Sweep & Frontier Pareto (BF16 Down to 1-Bit PTQ)

![Scuttle Quantization & Frontier Pareto](assets/scuttle_quant_frontier.png)

| Precision | Static VRAM | Peak VRAM (100-link) | 60-Flag CTF Solved | Retention | Latency / Run | Page AUC / F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **BF16** (`model.safetensors`, 3.46 GB) | 3.22 GiB | 3.99 GiB | 45 / 60 (75.0%) | 100.0% | 1,491 ms | 0.9546 / 0.7379 |
| **8-bit** (`bitsandbytes` LLM.int8) | 2.40 GiB | 3.89 GiB | 44 / 60 (73.3%) | 97.8% | 1,905 ms | 0.9565 / 0.7475 |
| **4-bit NF4** (`bitsandbytes`, BF16 emb) | 1.70 GiB | 3.02 GiB | 45 / 60 (75.0%) | 100.0% | 1,564 ms | 0.9534 / 0.7523 |
| **4-bit Packed + 4-bit Emb *(default, 973 MB)*** | **0.93 GiB** | **1.70 GiB** | **46 / 60 (76.7%)** | **~100%** *(102.2%)* | **1,423 ms** | **0.9545 / 0.7475** |
| **3-bit HQQ (g64) + 4-bit Emb** *(PTQ cliff)* | 0.76 GiB | 1.53 GiB | 36 / 60 (60.0%) | 80.0% | 2,508 ms | 0.9439 / 0.7071 |
| **Mixed 2-bit MLP + 4-bit Attn/Conv + 4-bit Emb** | 0.67 GiB | 1.48 GiB | 17 / 60 (28.3%) | 37.8% | 2,684 ms | 0.8421 / 0.0000 |
| **2-bit HQQ (g32 / g64) + 4-bit Emb** | 0.68 / 0.60 GiB | 1.43 / 1.34 GiB | 7 / 6 of 60 | 15.6 / 13.3% | 2,866 / 2,225 ms | 0.59 / 0.31 / 0.00 |
| **1.58-bit Ternary / 1-bit Binary (g64)** | 0.49 / 0.41 GiB | 1.25 / 1.14 GiB | 6 / 5 of 60 | 13.3 / 11.1% | 3,342 / 2,800 ms | 0.50 / 0.55 / 0.00 |

> **Why `4bit` packed + 4-bit embeddings is the default:** `bitsandbytes` leaves LFM3's 128k-vocab embedding table in BF16 (524 MB). Packing both all Linear layers and the embedding table in pure PyTorch (`scuttle_int4.safetensors`, **973 MB**) cuts static VRAM by **3.5×** (**3.22 → 0.93 GiB**), runs **faster than BF16**, requires **zero `bitsandbytes` / C++ extensions**, and hits the exact lossless Pareto knee right before the 3-bit PTQ cliff. Paired with `--idle-offload 120`, idle VRAM is **0.0 GiB**.

Read the full technical report in **[WHITEPAPER.md](WHITEPAPER.md)**.

---

## Installation

```bash
# Default install — includes 0.93 GiB pure-PyTorch 4-bit packed mode (no bitsandbytes needed!):
pip install "scuttle-search @ git+https://github.com/MLTQ/scuttle.git"
```

Pre-packed 4-bit weights (**973 MB**, `scuttle_int4.safetensors`) auto-download on first run from Hugging Face: [`Multiqueue/scuttle-1.7b`](https://huggingface.co/Multiqueue/scuttle-1.7b) (pass `--quant bf16` to fetch full 3.46 GB BF16 weights instead).

---

## Quickstart

### 1. Index any folder of Markdown, Obsidian, or HTML files

```bash
scuttle index ./my-notes-or-docs -o docs-graph.json.gz --domain notes --noun note
```

`scuttle index` automatically parses `[[wikilinks]]`, `[markdown](links.md)`, and HTML `<a href>` links along with the surrounding citation sentence.

Or grab ready-to-run real-world graphs (**US Federal Law eCFR** + **Wikispeedia** + **sample Markdown vault**) from the companion demo repo: **[`MLTQ/scuttle-demos`](https://github.com/MLTQ/scuttle-demos)**.

### 2. Run a Swarm Search from the CLI

```bash
scuttle search \
  --graph docs-graph.json.gz \
  --query "exemption from registration for private securities offerings to accredited investors" \
  --budget 16
```

### 3. Use as a Python Library

```python
from scuttle import load_world, ScuttleSearcher

world = load_world("ecfr-law-graph.json.gz")
searcher = ScuttleSearcher(world)  # defaults to Multiqueue/scuttle-1.7b in 0.93 GiB packed 4-bit

result = searcher.search(
    "exemption from registration for private securities offerings to accredited investors",
    n_seeds=4,
    budget=16,
)

for hit in result["contributing"]:
    trail = " -> ".join(t["title"] for t in hit["trail"])
    print(f"[{hit['p_contributes']:.3f}] {hit['title']}\n  Trail: {trail}")
```

---

## AI Agent Integration (MCP Server)

`scuttle-mcp` is a zero-bloat JSON-RPC 2.0 stdio MCP server that runs in **0.93 GiB static VRAM** by default and automatically evicts VRAM when idle (`--idle-offload`).

Drop this into your MCP config (**Antigravity**, **Claude Code**, **Cursor**, **Codex**, etc.):

```json
{
  "mcpServers": {
    "scuttle": {
      "command": "scuttle-mcp",
      "args": [
        "--idle-offload", "120",
        "--graphs", "/path/to/your/graphs"
      ]
    }
  }
}
```

**Exposed MCP tools:**
- `scuttle_search(graph, query, seeds=4, budget=16, start_ids=None, threshold=0.45)` — returns ranked contributing docs, calibrated `p_contributes`, exact hop-by-hop provenance trails, and latency.
- `scuttle_index_directory(directory, output_path, domain="docs", doc_noun="document")` — builds a Scuttle graph from local Markdown/HTML/text files on the fly.
- `scuttle_list_graphs()` — lists registered `.json.gz` graphs in `--graphs`.

---

## Custom Graph Schema (`graph.json.gz`)

Any corpus can be searched by Scuttle if exported to this minimal JSON (or gzipped JSON) format:

```json
{
  "domain": "my-corpus",
  "doc_noun": "document",
  "papers": {
    "doc_1": {
      "title": "Section 506 — Private Offerings",
      "abstract": "First ~700 chars of document text or summary..."
    }
  },
  "out": {
    "doc_1": [
      ["doc_2", "Optional sentence where doc_1 cites or links to doc_2."]
    ]
  }
}
```

Incoming edges are automatically inverted on load, and `ScuttleWorld` builds flexible up-to-100-link menus (`60 out + 40 in` with unused slot rollover).

---

## Licenses

- **Code (`MLTQ/scuttle`):** [MIT License](LICENSE).
- **Model Weights (`Multiqueue/scuttle-1.7b`):** Derived from Liquid AI's `LFM3-1.7B` and distributed under the **[LFM Open License v1.0](https://huggingface.co/Multiqueue/scuttle-1.7b/blob/main/LICENSE)** (free for research, non-commercial, and commercial use under \$10M annual revenue).
