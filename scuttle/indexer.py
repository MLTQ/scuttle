"""Index a local directory of Markdown, HTML, or text documents into a Scuttle graph.json.gz.

Supports:
  - Obsidian / Wiki links: [[Target Page]] or [[Target Page|alias]]
  - Standard Markdown links: [anchor text](relative/path.md)
  - HTML links: <a href="relative/path.html">anchor</a>
Extracts the surrounding context sentence for every resolved link automatically.
"""
import gzip
import json
import re
from pathlib import Path

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
MDLINK = re.compile(r"\[([^\]]+)\]\(([^)#\s]+)(?:#[^)\s]+)?\)")
HTMLLINK = re.compile(r'<a\s+[^>]*href=["\']([^"\'#]+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
HTML_TITLE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])")


def _clean(s):
    return re.sub(r"\s+", " ", s).strip()


def _strip_frontmatter(text):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :]
    return text


def index_directory(root_dir, output_path, domain="docs", doc_noun="document", extensions=(".md", ".markdown", ".txt", ".html", ".htm")):
    root = Path(root_dir).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions and not any(part.startswith(".") for part in p.relative_to(root).parts))

    papers = {}
    raw_texts = {}
    stem_to_ids = {}
    rel_to_id = {}

    for f in files:
        rel = f.relative_to(root).as_posix()
        doc_id = rel
        text = f.read_text(encoding="utf-8", errors="ignore")
        is_html = f.suffix.lower() in (".html", ".htm")
        if is_html:
            tm = HTML_TITLE.search(text)
            title = _clean(tm.group(1)) if tm else f.stem.replace("_", " ").replace("-", " ")
            plain = _clean(HTML_TAG.sub(" ", text))
        else:
            body = _strip_frontmatter(text)
            hm = HEADING.search(body)
            title = _clean(hm.group(1)) if hm else f.stem.replace("_", " ").replace("-", " ")
            plain = _clean(WIKILINK.sub(lambda m: m.group(2) or m.group(1), MDLINK.sub(r"\1", body)))

        if not plain:
            continue
        papers[doc_id] = dict(title=title, abstract=plain[:1200])
        raw_texts[doc_id] = (text, is_html)
        rel_to_id[rel.lower()] = doc_id
        rel_to_id[rel.rsplit(".", 1)[0].lower()] = doc_id
        stem_to_ids.setdefault(f.stem.lower(), []).append(doc_id)
        stem_to_ids.setdefault(title.lower(), []).append(doc_id)

    def resolve(src_id, target):
        t = target.strip()
        if not t or t.startswith(("http://", "https://", "mailto:")):
            return None
        # Try relative path resolution first
        src_parent = (root / src_id).parent
        cand_rel = (src_parent / t).resolve()
        try:
            r = cand_rel.relative_to(root).as_posix().lower()
            if r in rel_to_id:
                return rel_to_id[r]
        except ValueError:
            pass
        key = t.lstrip("/").lower()
        if key in rel_to_id:
            return rel_to_id[key]
        key_no_ext = key.rsplit(".", 1)[0]
        if key_no_ext in rel_to_id:
            return rel_to_id[key_no_ext]
        hits = stem_to_ids.get(Path(t).stem.lower()) or stem_to_ids.get(t.lower())
        return hits[0] if hits else None

    out = {}
    edge_count = 0
    for src_id, (raw, is_html) in raw_texts.items():
        first_ctx = {}
        order = []
        if is_html:
            for m in HTMLLINK.finditer(raw):
                dst = resolve(src_id, m.group(1))
                if dst and dst != src_id and dst not in first_ctx:
                    lo, hi = max(0, m.start() - 180), min(len(raw), m.end() + 180)
                    first_ctx[dst] = _clean(HTML_TAG.sub(" ", raw[lo:hi]))[:400]
                    order.append(dst)
        else:
            body = _strip_frontmatter(raw)
            for s in SENT_SPLIT.split(body):
                for m in WIKILINK.finditer(s):
                    dst = resolve(src_id, m.group(1))
                    if dst and dst != src_id and dst not in first_ctx:
                        ctx = _clean(WIKILINK.sub(lambda mm: mm.group(2) or mm.group(1), MDLINK.sub(r"\1", s)))[:400]
                        first_ctx[dst] = ctx
                        order.append(dst)
                for m in MDLINK.finditer(s):
                    dst = resolve(src_id, m.group(2))
                    if dst and dst != src_id and dst not in first_ctx:
                        ctx = _clean(WIKILINK.sub(lambda mm: mm.group(2) or mm.group(1), MDLINK.sub(r"\1", s)))[:400]
                        first_ctx[dst] = ctx
                        order.append(dst)
        if order:
            out[src_id] = [[dst, first_ctx[dst]] for dst in order]
            edge_count += len(order)

    graph = dict(domain=domain, doc_noun=doc_noun, papers=papers, out=out, questions=[])
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if out_p.suffix == ".gz" else open
    with op(out_p, "wt", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    return dict(output=str(out_p), documents=len(papers), edges=edge_count, domain=domain)
