"""Unified graph world with flexible 100-link menus for Scuttle."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

MAX_LINKS_DEFAULT = 100
MIN_OUT_DEFAULT = 60
MAX_D = 6


def stable_hash(*parts):
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x1f")
    return m.hexdigest()


class World:
    """Graph of linked documents (papers, articles, regulations, markdown notes, etc.).

    Flexible menu budget (default 100 links):
      1. Take up to `min_out` (60) outgoing links in reading order.
      2. Fill remaining slots up to `max_links` (100) with highest-indegree incoming links.
      3. If slots remain and the document has > `min_out` outgoing links, fill with remaining outgoing links.
    """

    def __init__(self, graph, max_links=MAX_LINKS_DEFAULT, min_out=MIN_OUT_DEFAULT):
        self.domain = graph.get("domain", "docs")
        self.doc_noun = graph.get("doc_noun", "document")
        self.papers = graph["papers"]
        self.questions = graph.get("questions", [])
        self.raw_out = {
            k: [(d, c) for d, c in v if d != k and d in self.papers]
            for k, v in graph.get("out", {}).items()
        }

        citers = defaultdict(list)
        self.uncapped_adj = defaultdict(set)
        for src, rows in self.raw_out.items():
            for d, c in rows:
                citers[d].append((src, c))
                self.uncapped_adj[src].add(d)
                self.uncapped_adj[d].add(src)
        self.citers = citers
        self.cited_count = Counter({k: len(v) for k, v in citers.items()})

        self.max_links = max_links
        self.min_out = min(min_out, max_links) if max_links else min_out
        self.menus = {}
        for k in self.papers:
            self.menus[k] = self._build_flex_menu(k, max_links=self.max_links, min_out=self.min_out)

    def _build_flex_menu(self, k, max_links=MAX_LINKS_DEFAULT, min_out=MIN_OUT_DEFAULT):
        outs = self.raw_out.get(k, [])
        ins = sorted(self.citers.get(k, []), key=lambda e: (-self.cited_count[e[0]], e[0]))
        menu, seen = [], {k}
        for d, c in outs:
            if max_links and len(menu) >= min_out:
                break
            if d not in seen:
                seen.add(d)
                menu.append((d, "cites", c))
        for s, c in ins:
            if max_links and len(menu) >= max_links:
                break
            if s not in seen:
                seen.add(s)
                menu.append((s, "cited by", c))
        if (not max_links or len(menu) < max_links) and len(outs) > min_out:
            for d, c in outs:
                if max_links and len(menu) >= max_links:
                    break
                if d not in seen:
                    seen.add(d)
                    menu.append((d, "cites", c))
        return menu

    def menu(self, page, hidden=()):
        if not hidden:
            return self.menus.get(page, [])
        return [m for m in self.menus.get(page, []) if m[0] not in hidden]

    def uncapped_distances(self, targets, hidden=(), max_d=MAX_D):
        dist = {t: 0 for t in targets if t in self.papers and t not in hidden}
        q = deque(dist)
        while q:
            x = q.popleft()
            if dist[x] >= max_d:
                continue
            for p in self.uncapped_adj[x]:
                if p not in dist and p not in hidden:
                    dist[p] = dist[x] + 1
                    q.append(p)
        return dist


def load_world(path, max_links=MAX_LINKS_DEFAULT, min_out=MIN_OUT_DEFAULT):
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return World(json.load(f), max_links=max_links, min_out=min_out)
