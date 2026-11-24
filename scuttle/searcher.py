"""ScuttleSearcher: Hybrid BM25-seeded or zero-index parallel swarm graph search."""
import math
import re
from collections import Counter

import torch

from scuttle.model import ABEL_W, DEFAULT_MODEL_ID, ScuttleWorker, frontier_value, snippet
from scuttle.world import stable_hash

TOKEN = re.compile(r"[a-z0-9]+")


class CorpusBM25:
    """Pure-Python inverted-index BM25 retriever over document titles & abstracts for fast auto-seeding."""

    def __init__(self, world):
        self.world = world
        self.postings = {}
        self.doc_len = {}
        total_len = 0
        for doc_id, p in world.papers.items():
            txt = (p.get("title") or "") + " " + (p.get("abstract") or "")
            toks = TOKEN.findall(txt.lower())
            if not toks:
                continue
            self.doc_len[doc_id] = len(toks)
            total_len += len(toks)
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((doc_id, tf))
        n = max(1, len(self.doc_len))
        self.avgdl = total_len / n
        self.idf = {t: math.log(1.0 + (n - len(ps) + 0.5) / (len(ps) + 0.5)) for t, ps in self.postings.items()}

    def search(self, query, k=16, exclude=()):
        q_terms = set(TOKEN.findall(query.lower()))
        scores = Counter()
        for t in q_terms:
            ps = self.postings.get(t)
            if not ps:
                continue
            idf = self.idf[t]
            for doc_id, tf in ps:
                if doc_id in exclude:
                    continue
                dl = self.doc_len[doc_id]
                norm = 1.2 * (1.0 - 0.75 + 0.75 * dl / self.avgdl)
                scores[doc_id] += idf * tf * 2.2 / (tf + norm)
        return [doc_id for doc_id, _ in scores.most_common(k)]


class ScuttleSearcher:
    """Parallel swarm graph searcher backed by a calibrated single-pass Scuttle worker."""

    def __init__(self, world, model=DEFAULT_MODEL_ID, device="auto", quant="4bit", weights=ABEL_W, worker=None):
        self.world = world
        self.weights = weights
        self.worker = worker or ScuttleWorker(model_id_or_path=model, device=device, quant=quant)
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = CorpusBM25(self.world)
        return self._bm25

    @torch.inference_mode()
    def _score_pages(self, query, pages, seen, hidden=()):
        menus = [self.world.menu(p, hidden) for p in pages]
        rows = [self.worker.menu_row(self.world, query, p, hidden, menu_override=m) for p, m in zip(pages, menus)]
        outs = self.worker.score_rows(rows)
        results = []
        for m, (pz, lz) in zip(menus, outs):
            p_contrib = float(torch.sigmoid(pz).item())
            links = {}
            if len(m) > 0:
                probs = torch.sigmoid(lz)
                vals = frontier_value(probs, weights=self.weights).cpu().tolist()
                p0s = probs[:, 0].cpu().tolist()
                for (dest, kind, _), v, p0 in zip(m, vals, p0s):
                    if dest not in seen:
                        links[dest] = dict(priority=float(v), p_hit=float(p0), kind=kind)
            results.append((p_contrib, links))
        return results

    def search(
        self,
        query,
        start_ids=None,
        n_seeds=4,
        budget=16,
        batch_k=2,
        contrib_threshold=0.45,
        hidden=(),
    ):
        """Run a parallel swarm search over the graph and return ranked contributing docs + provenance trails."""
        hidden_set = set(hidden)
        if start_ids:
            seeds = [s for s in start_ids if s in self.world.papers and s not in hidden_set]
        else:
            seeds = self.bm25.search(query, k=n_seeds, exclude=hidden_set)
        if not seeds:
            return dict(query=query, domain=self.world.domain, seeds=[], reads_used=0, contributing=[], top_pages=[])

        seeds = seeds[: min(len(seeds), budget)]
        seen = set(seeds)
        reads = list(seeds)
        parent = {s: None for s in seeds}
        pending = list(seeds)
        page_scores = {}
        frontier = {}  # dest -> max calibrated priority
        best_parent = {}

        while pending:
            scored = self._score_pages(query, pending, seen, hidden=hidden_set)
            for pg, (p_contrib, links) in zip(pending, scored):
                page_scores[pg] = p_contrib
                for dest, info in links.items():
                    if dest not in seen and info["priority"] > frontier.get(dest, -math.inf):
                        frontier[dest] = info["priority"]
                        best_parent[dest] = pg
            pending = []
            if len(reads) >= budget or not frontier:
                break
            take = min(batch_k, budget - len(reads), len(frontier))
            for _ in range(take):
                dest = max(frontier, key=lambda d: (frontier[d], stable_hash("scuttle-tie", query, d)))
                del frontier[dest]
                seen.add(dest)
                reads.append(dest)
                parent[dest] = best_parent.get(dest)
                pending.append(dest)

        def trail_of(doc_id):
            path, cur = [], doc_id
            while cur is not None and len(path) < 32:
                path.append(cur)
                cur = parent.get(cur)
            return list(reversed(path))

        ranked = sorted(reads, key=lambda d: page_scores.get(d, 0.0), reverse=True)
        formatted = [
            dict(
                doc_id=d,
                title=self.world.papers[d].get("title", ""),
                p_contributes=round(page_scores.get(d, 0.0), 4),
                read_step=reads.index(d) + 1,
                from_seed=d in seeds,
                trail=[dict(doc_id=t, title=self.world.papers[t].get("title", "")) for t in trail_of(d)],
                snippet=snippet(self.world.papers[d].get("abstract", ""), max_sentences=2, max_chars=300),
            )
            for d in ranked
        ]
        contributing = [x for x in formatted if x["p_contributes"] >= contrib_threshold]
        return dict(
            query=query,
            domain=self.world.domain,
            budget=budget,
            reads_used=len(reads),
            seeds=seeds,
            contributing=contributing,
            top_pages=formatted[:8],
            all_read_ids=reads,
        )
