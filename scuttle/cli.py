"""Scuttle CLI: `scuttle search` and `scuttle index`."""
import argparse
import json
import time

from scuttle.indexer import index_directory
from scuttle.model import DEFAULT_MODEL_ID
from scuttle.searcher import ScuttleSearcher
from scuttle.world import load_world


def cmd_search(a):
    t0 = time.perf_counter()
    world = load_world(a.graph, max_links=a.max_links)
    searcher = ScuttleSearcher(world, model=a.model, device=a.device, quant=a.quant)
    starts = [x.strip() for x in a.start_ids.split(",") if x.strip()] if a.start_ids else None
    t_s = time.perf_counter()
    res = searcher.search(
        a.query,
        start_ids=starts,
        n_seeds=a.seeds,
        budget=a.budget,
        batch_k=a.batch_k,
        contrib_threshold=a.threshold,
    )
    res["search_ms"] = round(1000 * (time.perf_counter() - t_s), 1)
    res["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(f"Query: {res['query']}")
        print(f"Domain: {res['domain']} | Reads: {res['reads_used']}/{res['budget']} | Search: {res['search_ms']:.0f} ms")
        print(f"\nSurfaced contributing documents (>= {a.threshold}): {len(res['contributing'])}")
        for item in (res["contributing"] or res["top_pages"][:5]):
            tag = "SEED" if item["from_seed"] else f"HOP (step {item['read_step']})"
            print(f"  [{item['p_contributes']:.3f}] [{tag}] {item['title']} ({item['doc_id']})")
            if len(item["trail"]) > 1:
                print("     Trail: " + " -> ".join(t["title"][:45] for t in item["trail"]))


def cmd_index(a):
    info = index_directory(a.directory, a.output, domain=a.domain, doc_noun=a.noun)
    print(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="scuttle", description="Scuttle: fast frontier-calibrated swarm graph search")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Run a Scuttle parallel swarm search on a graph.json.gz")
    s.add_argument("--graph", required=True, help="Path to graph.json or graph.json.gz")
    s.add_argument("--query", required=True)
    s.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF repo ID or local checkpoint dir")
    s.add_argument("--device", default="auto", help="auto|cuda|cuda:0|cpu")
    s.add_argument("--quant", choices=["4bit", "bf16", "none", "bnb4", "8bit"], default="4bit",
                   help="4bit (default: 0.93 GiB pure-PyTorch packed + 4-bit emb) | bf16 | bnb4 | 8bit")
    s.add_argument("--start-ids", help="Comma-separated starting doc IDs (omit for BM25 auto-seeding)")
    s.add_argument("--seeds", type=int, default=4)
    s.add_argument("--budget", type=int, default=16)
    s.add_argument("--batch-k", type=int, default=2)
    s.add_argument("--max-links", type=int, default=100)
    s.add_argument("--threshold", type=float, default=0.45)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    ix = sub.add_parser("index", help="Build a Scuttle graph.json.gz from a folder of Markdown/HTML/text files")
    ix.add_argument("directory", help="Directory to index")
    ix.add_argument("-o", "--output", required=True, help="Output .json.gz path")
    ix.add_argument("--domain", default="docs")
    ix.add_argument("--noun", default="document", help="Document noun (document|paper|article|regulation|note)")
    ix.set_defaults(func=cmd_index)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
