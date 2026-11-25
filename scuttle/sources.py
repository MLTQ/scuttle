"""Universal local source -> Scuttle graph builder with incremental disk caching.

Turns any of the following into a navigable Scuttle `graph.json.gz` (with full `path` & `kind` metadata):
  1. Markdown / Obsidian piles (.md, .markdown, .txt, .rst, .org, .html) — [[wikilinks]], [md](links),
     HTML <a href>, YAML frontmatter title/aliases, image embeds.
  2. Messy dataswamps (mixed folders of PDFs via pdftotext/pypdf, Markdown, HTML, code files with
     import/include edges, JSON/JSONL/CSV, and image stubs carrying exact paths for follow-up vision).
  3. User SQLite databases (.db, .sqlite, .sqlite3, or sqlite:///path#table1,table2) — opened strictly
     READ-ONLY (`mode=ro`), auto-introspecting tables, text columns, foreign-key edges (`PRAGMA
     foreign_key_list`), heuristic `*_id` links, and inline cross-references.
  4. Structured tabular/record files (.jsonl, .ndjson, .csv, .tsv, or array .json).
  5. Prebuilt Scuttle graphs (.json, .json.gz).

Documents with few explicit links automatically receive lightweight BM25/TF-IDF lexical neighbor
bridges (capped to avoid mega-hubs) so even an unlinked pile of PDFs or flat DB rows is scuttlable.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scuttle.world import World

WIKILINK = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
MDLINK = re.compile(r"!?\[([^\]]*)\]\(([^)#\s]+)(?:#[^)\s]+)?\)")
HTMLLINK = re.compile(r'<a\s+[^>]*href=["\']([^"\'#]+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
HTML_IMG = re.compile(r'<img\s+[^>]*src=["\']([^"\'#]+)["\'][^>]*>', re.IGNORECASE)
HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
HTML_TITLE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])")
TOKEN = re.compile(r"[a-z0-9_]{2,}")

# Code import/include link extractors
PY_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
JS_IMPORT = re.compile(r"""(?:from|require\s*\()\s*['"](\.[^'"]+)['"]""")
C_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".next", ".nuxt", ".cache",
    ".obsidian", ".trash", ".idea", ".vscode", ".tox", "site-packages",
}

TEXT_EXTS = {".md", ".markdown", ".mdx", ".txt", ".rst", ".org", ".tex", ".log"}
HTML_EXTS = {".html", ".htm"}
CODE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rs", ".go", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".java", ".kt", ".rb", ".php", ".sql", ".sh", ".bash", ".zsh", ".yaml",
    ".yml", ".toml", ".ini", ".cfg",
}
DATA_EXTS = {".jsonl", ".ndjson", ".csv", ".tsv", ".json"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".bmp", ".tiff"}
SQLITE_EXTS = {".db", ".sqlite", ".sqlite3"}

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have", "has",
    "had", "not", "but", "you", "your", "can", "will", "all", "any", "one", "two", "use", "used",
    "using", "into", "out", "about", "more", "when", "what", "which", "who", "how", "why", "its",
    "their", "there", "these", "those", "then", "than", "also", "been", "being", "some", "such",
    "only", "other", "each", "over", "under", "between", "through", "where", "while", "should",
    "would", "could", "may", "might", "must", "shall", "true", "false", "none", "null", "def",
    "return", "import", "class", "self", "const", "let", "var", "function", "async", "await",
}

TITLE_COL_PRIORITY = (
    "title", "name", "subject", "heading", "label", "slug", "summary", "topic",
    "filename", "file_name", "path", "key", "code", "email", "username",
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Minimal zero-dependency YAML frontmatter parser (title, aliases, tags)."""
    meta: dict[str, Any] = {}
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return meta, text
    end = text.find("\n---", 4)
    if end == -1:
        return meta, text
    fm_block = text[4:end]
    body = text[end + 4 :].lstrip("\r\n")
    cur_list_key = None
    for raw_line in fm_block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and cur_list_key:
            meta.setdefault(cur_list_key, []).append(line[2:].strip().strip("\"'"))
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            cur_list_key = None
            if not v:
                cur_list_key = k
                meta.setdefault(k, [])
            elif v.startswith("[") and v.endswith("]"):
                items = [x.strip().strip("\"'") for x in v[1:-1].split(",") if x.strip()]
                meta[k] = items
            else:
                meta[k] = v.strip("\"'")
    return meta, body


def _extract_pdf_text(path: Path, max_chars: int = 16000) -> str:
    """Extract text from PDF using `pdftotext` (preferred) or `pypdf` if available."""
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        try:
            proc = subprocess.run(
                [pdftotext, "-q", "-l", "25", "-nopgbrk", str(path), "-"],
                capture_output=True, text=True, timeout=15, errors="ignore",
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout[:max_chars]
        except Exception:
            pass
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        chunks = []
        tot = 0
        for page in reader.pages[:25]:
            txt = page.extract_text() or ""
            chunks.append(txt)
            tot += len(txt)
            if tot >= max_chars:
                break
        return "\n".join(chunks)[:max_chars]
    except Exception:
        return ""


def _add_lexical_bridges(
    papers: dict[str, dict[str, Any]],
    out: dict[str, list[list[str]]],
    min_explicit: int = 6,
    bridge_k: int = 6,
    max_in_bridge: int = 18,
) -> int:
    """Add BM25-style nearest-neighbor edges ONLY for low-degree docs, avoiding mega-hubs."""
    if bridge_k <= 0 or len(papers) < 2:
        return 0

    doc_tokens: dict[str, Counter] = {}
    df: Counter = Counter()
    for did, p in papers.items():
        if p.get("kind") == "image":
            continue
        txt = f"{p.get('title', '')} {p.get('abstract', '')}"
        toks = [t for t in TOKEN.findall(txt.lower()) if t not in STOPWORDS and not t.isdigit()]
        if len(toks) < 3:
            continue
        c = Counter(toks)
        doc_tokens[did] = c
        for term in c:
            df[term] += 1

    n_docs = len(doc_tokens)
    if n_docs < 2:
        return 0

    # Skip overly common terms (>35% of corpus) to avoid generic mega-hubs
    max_df = max(2, int(0.35 * n_docs))
    postings: dict[str, list[tuple[str, float]]] = {}
    doc_norm: dict[str, float] = {}
    idf: dict[str, float] = {}

    for term, count in df.items():
        if count > max_df and n_docs >= 20:
            continue
        w_idf = math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
        idf[term] = w_idf

    for did, c in doc_tokens.items():
        sq = 0.0
        for term, tf in c.items():
            if term not in idf:
                continue
            w = (1.0 + math.log(tf)) * idf[term]
            sq += w * w
        norm = math.sqrt(sq) or 1.0
        doc_norm[did] = norm
        for term, tf in c.items():
            if term not in idf:
                continue
            w = (1.0 + math.log(tf)) * idf[term] / norm
            postings.setdefault(term, []).append((did, w))

    # Track incoming bridge counts so no single doc turns into a synthetic mega-hub
    in_bridge: Counter = Counter()
    added = 0

    for did, c in doc_tokens.items():
        existing = {dst for dst, _ in out.get(did, ())}
        if len(existing) >= min_explicit:
            continue
        need = min(bridge_k, max(0, min_explicit - len(existing)))
        if need <= 0:
            continue

        scores: Counter = Counter()
        shared_terms: dict[str, list[tuple[float, str]]] = {}
        norm = doc_norm[did]
        # Score using top 40 highest-weight terms of this doc for speed on big corpora
        top_terms = sorted(
            (( (1.0 + math.log(tf)) * idf[t] / norm, t ) for t, tf in c.items() if t in idf),
            reverse=True,
        )[:40]

        for q_w, term in top_terms:
            for other, o_w in postings.get(term, ()):
                if other == did or other in existing:
                    continue
                contrib = q_w * o_w
                scores[other] += contrib
                shared_terms.setdefault(other, []).append((contrib, term))

        if not scores:
            continue
        edges = out.setdefault(did, [])
        for other, sim in scores.most_common(need * 3):
            if sim < 0.08:
                break
            if in_bridge[other] >= max_in_bridge:
                continue
            top_shared = [t for _, t in sorted(shared_terms[other], reverse=True)[:6]]
            ctx = f"Related via shared terms ({', '.join(top_shared)}): {papers[other].get('title', other)}"
            edges.append([other, ctx[:320]])
            in_bridge[other] += 1
            added += 1
            if len(edges) >= min_explicit:
                break

    return added


# ==============================================================================
# 1. Directory / Markdown Pile / Messy Dataswamp Indexer
# ==============================================================================

def index_dataswamp(
    root_dir: str | Path,
    output_path: str | Path | None = None,
    domain: str = "dataswamp",
    doc_noun: str = "document",
    max_files: int = 10000,
    bridge_k: int = 6,
    include_code: bool = True,
    include_pdfs: bool = True,
    include_images: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Index a folder of Markdown, Obsidian notes, PDFs, HTML, code, JSONL/CSV, and image stubs."""
    t0 = time.perf_counter()
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    allowed = set(TEXT_EXTS) | set(HTML_EXTS) | set(DATA_EXTS)
    if include_code:
        allowed |= set(CODE_EXTS)
    if include_pdfs:
        allowed |= set(PDF_EXTS)
    if include_images:
        allowed |= set(IMAGE_EXTS)

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() in allowed:
                files.append(p)
                if len(files) >= max_files:
                    break
        if len(files) >= max_files:
            break

    papers: dict[str, dict[str, Any]] = {}
    raw_meta: dict[str, tuple[str, str]] = {}  # doc_id -> (raw_text, kind)
    rel_to_id: dict[str, str] = {}
    stem_to_ids: dict[str, list[str]] = {}
    counts_by_kind: Counter = Counter()

    for f in files:
        rel = f.relative_to(root).as_posix()
        ext = f.suffix.lower()
        doc_id = rel

        if ext in IMAGE_EXTS:
            title = f.stem.replace("_", " ").replace("-", " ")
            papers[doc_id] = dict(
                title=f"[Image] {title}",
                abstract=f"Image file at {f} ({f.name}). Inspect with a vision tool if relevant.",
                path=str(f),
                kind="image",
            )
            rel_to_id[rel.lower()] = doc_id
            stem_to_ids.setdefault(f.name.lower(), []).append(doc_id)
            stem_to_ids.setdefault(f.stem.lower(), []).append(doc_id)
            counts_by_kind["image"] += 1
            continue

        if ext in PDF_EXTS:
            raw = _extract_pdf_text(f)
            if not raw.strip():
                continue
            first_lines = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) >= 4][:4]
            title = _clean(first_lines[0])[:110] if first_lines else f.stem.replace("_", " ").replace("-", " ")
            plain = _clean(raw)
            papers[doc_id] = dict(title=title, abstract=plain[:1800], path=str(f), kind="pdf")
            raw_meta[doc_id] = (raw, "pdf")
            counts_by_kind["pdf"] += 1
        elif ext in HTML_EXTS:
            raw = f.read_text(encoding="utf-8", errors="ignore")[:64000]
            tm = HTML_TITLE.search(raw)
            title = _clean(tm.group(1)) if tm else f.stem.replace("_", " ").replace("-", " ")
            plain = _clean(HTML_TAG.sub(" ", raw))
            if not plain:
                continue
            papers[doc_id] = dict(title=title, abstract=plain[:1800], path=str(f), kind="html")
            raw_meta[doc_id] = (raw, "html")
            counts_by_kind["html"] += 1
        elif ext in CODE_EXTS:
            raw = f.read_text(encoding="utf-8", errors="ignore")[:48000]
            plain = _clean(raw)
            if not plain:
                continue
            papers[doc_id] = dict(title=rel, abstract=plain[:1800], path=str(f), kind="code")
            raw_meta[doc_id] = (raw, "code")
            counts_by_kind["code"] += 1
        elif ext in DATA_EXTS:
            raw = f.read_text(encoding="utf-8", errors="ignore")[:32000]
            plain = _clean(raw)
            if not plain:
                continue
            papers[doc_id] = dict(title=rel, abstract=plain[:1800], path=str(f), kind="data")
            raw_meta[doc_id] = (raw, "data")
            counts_by_kind["data"] += 1
        else:
            raw = f.read_text(encoding="utf-8", errors="ignore")[:64000]
            fm, body = _parse_frontmatter(raw)
            hm = HEADING.search(body)
            title = _clean(str(fm.get("title") or (hm.group(1) if hm else f.stem.replace("_", " ").replace("-", " "))))
            plain = _clean(WIKILINK.sub(lambda m: m.group(2) or m.group(1), MDLINK.sub(r"\1", body)))
            if not plain:
                continue
            tags = fm.get("tags")
            if isinstance(tags, list) and tags:
                plain = f"Tags: {', '.join(str(x) for x in tags[:10])}. {plain}"
            papers[doc_id] = dict(title=title, abstract=plain[:1800], path=str(f), kind="markdown")
            raw_meta[doc_id] = (body, "markdown")
            for alias in (fm.get("aliases") if isinstance(fm.get("aliases"), list) else []):
                stem_to_ids.setdefault(str(alias).lower(), []).append(doc_id)
            counts_by_kind["markdown"] += 1

        rel_to_id[rel.lower()] = doc_id
        rel_to_id[rel.rsplit(".", 1)[0].lower()] = doc_id
        stem_to_ids.setdefault(f.name.lower(), []).append(doc_id)
        stem_to_ids.setdefault(f.stem.lower(), []).append(doc_id)
        stem_to_ids.setdefault(papers[doc_id]["title"].lower(), []).append(doc_id)

    def resolve(src_id: str, target: str) -> str | None:
        t = target.strip().split("?")[0]
        if not t or t.startswith(("http://", "https://", "mailto:", "#")):
            return None
        src_parent = (root / src_id).parent
        try:
            cand_rel = (src_parent / t).resolve().relative_to(root).as_posix().lower()
            if cand_rel in rel_to_id:
                return rel_to_id[cand_rel]
            cand_no_ext = cand_rel.rsplit(".", 1)[0]
            if cand_no_ext in rel_to_id:
                return rel_to_id[cand_no_ext]
        except Exception:
            pass
        key = t.lstrip("./").lstrip("/").lower()
        if key in rel_to_id:
            return rel_to_id[key]
        key_no_ext = key.rsplit(".", 1)[0]
        if key_no_ext in rel_to_id:
            return rel_to_id[key_no_ext]
        hits = stem_to_ids.get(Path(t).name.lower()) or stem_to_ids.get(Path(t).stem.lower()) or stem_to_ids.get(t.lower())
        return hits[0] if hits else None

    out: dict[str, list[list[str]]] = {}
    explicit_edges = 0

    for src_id, (raw, kind) in raw_meta.items():
        first_ctx: dict[str, str] = {}
        order: list[str] = []

        def add_edge(dst: str | None, ctx: str):
            if dst and dst != src_id and dst not in first_ctx:
                first_ctx[dst] = _clean(ctx)[:360]
                order.append(dst)

        if kind == "html":
            for m in HTMLLINK.finditer(raw):
                lo, hi = max(0, m.start() - 160), min(len(raw), m.end() + 160)
                add_edge(resolve(src_id, m.group(1)), HTML_TAG.sub(" ", raw[lo:hi]))
            for m in HTML_IMG.finditer(raw):
                add_edge(resolve(src_id, m.group(1)), f"Embedded image {m.group(1)}")
        elif kind == "code":
            for m in PY_IMPORT.finditer(raw):
                mod = (m.group(1) or m.group(2) or "").replace(".", "/")
                dst = resolve(src_id, mod) or resolve(src_id, mod + ".py") or resolve(src_id, mod + "/__init__.py")
                add_edge(dst, m.group(0))
            for m in JS_IMPORT.finditer(raw):
                add_edge(resolve(src_id, m.group(1)), m.group(0))
            for m in C_INCLUDE.finditer(raw):
                add_edge(resolve(src_id, m.group(1)), m.group(0))
        else:
            for s in SENT_SPLIT.split(raw):
                for m in WIKILINK.finditer(s):
                    ctx = WIKILINK.sub(lambda mm: mm.group(2) or mm.group(1), MDLINK.sub(r"\1", s))
                    add_edge(resolve(src_id, m.group(1)), ctx)
                for m in MDLINK.finditer(s):
                    ctx = WIKILINK.sub(lambda mm: mm.group(2) or mm.group(1), MDLINK.sub(r"\1", s))
                    add_edge(resolve(src_id, m.group(2)), ctx)

        if order:
            out[src_id] = [[dst, first_ctx[dst]] for dst in order]
            explicit_edges += len(order)

    bridge_edges = _add_lexical_bridges(papers, out, min_explicit=max(4, bridge_k), bridge_k=bridge_k)
    graph = dict(domain=domain, doc_noun=doc_noun, papers=papers, out=out, questions=[])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        op = gzip.open if out_p.suffix == ".gz" else open
        with op(out_p, "wt", encoding="utf-8") as fh:
            json.dump(graph, fh, ensure_ascii=False)

    info = dict(
        source=str(root),
        source_type="dataswamp",
        output=str(output_path) if output_path else None,
        documents=len(papers),
        explicit_edges=explicit_edges,
        bridge_edges=bridge_edges,
        edges=explicit_edges + bridge_edges,
        kinds=dict(counts_by_kind),
        domain=domain,
        seconds=round(time.perf_counter() - t0, 2),
    )
    return graph, info


# ==============================================================================
# 2. Read-Only SQLite Database Indexer (Foreign-Key + Text Graph!)
# ==============================================================================

def _is_internal_table(name: str) -> bool:
    low = name.lower()
    if low.startswith("sqlite_"):
        return True
    for suf in ("_fts", "_fts_data", "_fts_idx", "_fts_docsize", "_fts_config", "_fts_content",
                "_content", "_segments", "_segdir", "_docsize", "_stat"):
        if low.endswith(suf):
            return True
    return False


def index_sqlite(
    db_path: str | Path,
    output_path: str | Path | None = None,
    tables: list[str] | None = None,
    domain: str | None = None,
    doc_noun: str = "record",
    max_rows_per_table: int = 4000,
    max_total_docs: int = 20000,
    bridge_k: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open any SQLite database strictly READ-ONLY and build a foreign-key + text Scuttle graph."""
    t0 = time.perf_counter()
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        all_tables = [
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            if not _is_internal_table(r[0])
        ]
        chosen = [t for t in all_tables if t in set(tables)] if tables else all_tables
        if not chosen:
            raise ValueError(f"No readable user tables found in {path} (available: {all_tables})")

        # Map lowercase/singular table names for heuristic *_id links
        table_lookup: dict[str, str] = {}
        for t in chosen:
            tl = t.lower()
            table_lookup[tl] = t
            if tl.endswith("s") and len(tl) > 2:
                table_lookup[tl[:-1]] = t
            if tl.endswith("ies") and len(tl) > 4:
                table_lookup[tl[:-3] + "y"] = t

        papers: dict[str, dict[str, Any]] = {}
        pending_fks: list[tuple[str, str, str, str]] = []  # (src_doc_id, ref_table, ref_val, col_name)
        table_counts: dict[str, int] = {}

        for tbl in chosen:
            if len(papers) >= max_total_docs:
                break
            cols_info = conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            if not cols_info:
                continue
            col_names = [c["name"] for c in cols_info]
            pk_cols = [c["name"] for c in cols_info if c["pk"]]
            pk_col = pk_cols[0] if pk_cols else None

            # Declared foreign keys
            fks: dict[str, str] = {}
            for fk in conn.execute(f'PRAGMA foreign_key_list("{tbl}")').fetchall():
                if fk["table"] in chosen:
                    fks[fk["from"]] = fk["table"]
            # Heuristic *_id foreign keys
            for cn in col_names:
                if cn not in fks and cn.lower().endswith("_id") and len(cn) > 3:
                    stem = cn[:-3].lower()
                    if stem in table_lookup:
                        fks[cn] = table_lookup[stem]

            # Choose best title column
            title_col = None
            col_lower_map = {c.lower(): c for c in col_names}
            for cand in TITLE_COL_PRIORITY:
                if cand in col_lower_map:
                    title_col = col_lower_map[cand]
                    break

            limit = min(max_rows_per_table, max_total_docs - len(papers))
            try:
                rows = conn.execute(f'SELECT rowid AS _scuttle_rowid, * FROM "{tbl}" ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(f'SELECT * FROM "{tbl}" LIMIT ?', (limit,)).fetchall()

            kept = 0
            for idx, row in enumerate(rows):
                keys = row.keys()
                pk_val = row[pk_col] if (pk_col and pk_col in keys and row[pk_col] is not None) else (
                    row["_scuttle_rowid"] if "_scuttle_rowid" in keys else idx
                )
                doc_id = f"{tbl}:{pk_val}"
                parts = []
                for cn in col_names:
                    v = row[cn]
                    if v is None or isinstance(v, (bytes, bytearray)):
                        continue
                    sv = _clean(str(v))
                    if not sv:
                        continue
                    parts.append(f"{cn}: {sv[:500]}")
                abstract = " | ".join(parts)[:1800]
                if len(abstract) < 8:
                    continue

                raw_title = _clean(str(row[title_col]))[:100] if (title_col and row[title_col] is not None) else ""
                title = f"[{tbl}] {raw_title}" if raw_title else f"{tbl} #{pk_val}"
                papers[doc_id] = dict(
                    title=title,
                    abstract=abstract,
                    path=f"sqlite://{path}#{tbl}:{pk_val}",
                    kind="sqlite_row",
                    table=tbl,
                )
                kept += 1

                for fk_col, ref_tbl in fks.items():
                    fv = row[fk_col]
                    if fv is not None and str(fv).strip():
                        pending_fks.append((doc_id, ref_tbl, str(fv).strip(), fk_col))

            if kept:
                table_counts[tbl] = kept
    finally:
        conn.close()

    out: dict[str, list[list[str]]] = {}
    explicit_edges = 0
    seen_edges: set[tuple[str, str]] = set()
    for src_id, ref_tbl, ref_val, col_name in pending_fks:
        dst_id = f"{ref_tbl}:{ref_val}"
        if dst_id in papers and dst_id != src_id and (src_id, dst_id) not in seen_edges:
            seen_edges.add((src_id, dst_id))
            ctx = f"Foreign key {col_name} -> {papers[dst_id]['title']}"
            out.setdefault(src_id, []).append([dst_id, ctx[:320]])
            explicit_edges += 1

    bridge_edges = _add_lexical_bridges(papers, out, min_explicit=max(4, bridge_k), bridge_k=bridge_k)
    dom = domain or f"sqlite:{path.stem}"
    graph = dict(domain=dom, doc_noun=doc_noun, papers=papers, out=out, questions=[])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        op = gzip.open if out_p.suffix == ".gz" else open
        with op(out_p, "wt", encoding="utf-8") as fh:
            json.dump(graph, fh, ensure_ascii=False)

    info = dict(
        source=str(path),
        source_type="sqlite",
        output=str(output_path) if output_path else None,
        documents=len(papers),
        explicit_edges=explicit_edges,
        bridge_edges=bridge_edges,
        edges=explicit_edges + bridge_edges,
        tables=table_counts,
        domain=dom,
        seconds=round(time.perf_counter() - t0, 2),
    )
    return graph, info


# ==============================================================================
# 3. Structured File Indexer (JSONL / NDJSON / CSV / TSV / JSON array)
# ==============================================================================

def index_structured_file(
    file_path: str | Path,
    output_path: str | Path | None = None,
    domain: str | None = None,
    doc_noun: str = "record",
    max_rows: int = 20000,
    bridge_k: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    path = Path(file_path).expanduser().resolve()
    ext = path.suffix.lower()
    records: list[dict[str, Any]] = []

    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh, delimiter=delim)):
                if i >= max_rows:
                    break
                records.append(dict(row))
    elif ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
    elif ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(obj, list):
            records = [x for x in obj[:max_rows] if isinstance(x, dict)]

    papers: dict[str, dict[str, Any]] = {}
    out: dict[str, list[list[str]]] = {}
    id_set: set[str] = set()

    for i, rec in enumerate(records):
        did = str(rec.get("id") or rec.get("doc_id") or rec.get("key") or f"row_{i}")
        title = ""
        for cand in TITLE_COL_PRIORITY:
            if rec.get(cand):
                title = _clean(str(rec[cand]))[:110]
                break
        if not title:
            title = f"{path.stem} #{did}"
        parts = [f"{k}: {_clean(str(v))[:450]}" for k, v in rec.items() if v is not None and _clean(str(v))]
        abstract = " | ".join(parts)[:1800]
        if not abstract:
            continue
        papers[did] = dict(title=title, abstract=abstract, path=f"{path}#{did}", kind=ext.lstrip("."))
        id_set.add(did)

    # Link explicit `links` / `references` / `parent_id` / `*_id` fields if present
    explicit_edges = 0
    for i, rec in enumerate(records):
        src = str(rec.get("id") or rec.get("doc_id") or rec.get("key") or f"row_{i}")
        if src not in papers:
            continue
        seen_dst: set[str] = set()
        for k, v in rec.items():
            kl = k.lower()
            cands = []
            if kl in ("links", "references", "refs", "citations", "children", "related") and isinstance(v, list):
                cands = [str(x) for x in v]
            elif kl.endswith("_id") and v is not None:
                cands = [str(v)]
            for dst in cands:
                if dst in id_set and dst != src and dst not in seen_dst:
                    seen_dst.add(dst)
                    out.setdefault(src, []).append([dst, f"Field {k} -> {papers[dst]['title']}"])
                    explicit_edges += 1

    bridge_edges = _add_lexical_bridges(papers, out, min_explicit=max(4, bridge_k), bridge_k=bridge_k)
    dom = domain or path.stem
    graph = dict(domain=dom, doc_noun=doc_noun, papers=papers, out=out, questions=[])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        op = gzip.open if out_p.suffix == ".gz" else open
        with op(out_p, "wt", encoding="utf-8") as fh:
            json.dump(graph, fh, ensure_ascii=False)

    info = dict(
        source=str(path),
        source_type=ext.lstrip("."),
        output=str(output_path) if output_path else None,
        documents=len(papers),
        explicit_edges=explicit_edges,
        bridge_edges=bridge_edges,
        edges=explicit_edges + bridge_edges,
        domain=dom,
        seconds=round(time.perf_counter() - t0, 2),
    )
    return graph, info


# ==============================================================================
# 4. Unified Fingerprint Cache & Source Resolver
# ==============================================================================

def _strip_prefix(spec: str) -> tuple[str, list[str] | None]:
    """Accept `swamp:/path`, `sqlite:///path#t1,t2`, or plain `/path`."""
    s = spec.strip()
    tables = None
    if "#" in s and not Path(s).exists():
        base, frag = s.rsplit("#", 1)
        if frag.strip():
            tables = [x.strip() for x in frag.split(",") if x.strip()]
        s = base
    for pref in ("swamp:", "dir:", "folder:", "sqlite://", "sqlite:", "db:"):
        if s.startswith(pref):
            s = s[len(pref) :]
            break
    return s, tables


def source_fingerprint(path: Path, tables: list[str] | None = None, max_files: int = 10000) -> str:
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode())
    if tables:
        h.update(("::" + ",".join(sorted(tables))).encode())
    if path.is_file():
        st = path.stat()
        h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
        # Also include SQLite WAL mtime if present
        wal = Path(str(path) + "-wal")
        if wal.exists():
            wst = wal.stat()
            h.update(f"wal:{wst.st_mtime_ns}:{wst.st_size}".encode())
    else:
        n = 0
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                fp = Path(dirpath) / fn
                try:
                    st = fp.stat()
                    h.update(f"{fp.relative_to(path)}:{st.st_mtime_ns}:{st.st_size}\n".encode())
                except OSError:
                    continue
                n += 1
                if n >= max_files:
                    break
            if n >= max_files:
                break
    return h.hexdigest()[:16]


def load_or_build_source(
    source_spec: str,
    cache_dir: str | Path,
    graphs_dir: str | Path | None = None,
    max_links: int = 100,
    refresh: bool = False,
    tables: list[str] | None = None,
    max_files: int = 10000,
    bridge_k: int = 6,
) -> tuple[World, dict[str, Any]]:
    """Resolve any named graph, prebuilt .json.gz, directory/dataswamp, SQLite DB, or CSV/JSONL."""
    # 1. Check pre-registered named graphs first
    if graphs_dir:
        gd = Path(graphs_dir).expanduser()
        if gd.exists():
            for p in sorted(gd.glob("*.json*")):
                name = p.name.replace("-graph.json.gz", "").replace(".json.gz", "").replace(".json", "")
                if source_spec == name:
                    source_spec = str(p)
                    break

    raw_path, frag_tables = _strip_prefix(source_spec)
    tables = tables or frag_tables
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source_spec} (resolved: {path})")

    # 2. Prebuilt Scuttle graph.json / graph.json.gz
    if path.is_file() and (path.name.endswith(".json.gz") or (path.suffix == ".json" and _is_scuttle_graph(path))):
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        w = World(data, max_links=max_links)
        n_edges = sum(len(v) for v in w.out.values())
        return w, dict(source=str(path), source_type="graph", cached=True, documents=len(w.papers), edges=n_edges, domain=w.domain)

    # 3. Cached build for directory / SQLite / CSV / JSONL
    cdir = Path(cache_dir).expanduser().resolve()
    cdir.mkdir(parents=True, exist_ok=True)
    fp = source_fingerprint(path, tables=tables, max_files=max_files)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.name)[:48]
    cache_file = cdir / f"{slug}-{fp}.json.gz"
    meta_file = cdir / f"{slug}-{fp}.meta.json"

    if cache_file.exists() and not refresh:
        with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        w = World(data, max_links=max_links)
        info = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        info.update(cached=True, cache_file=str(cache_file), documents=len(w.papers))
        return w, info

    if path.is_dir():
        graph, info = index_dataswamp(path, output_path=cache_file, max_files=max_files, bridge_k=bridge_k)
    elif path.suffix.lower() in SQLITE_EXTS or _looks_like_sqlite(path):
        graph, info = index_sqlite(path, output_path=cache_file, tables=tables, bridge_k=bridge_k)
    elif path.suffix.lower() in DATA_EXTS:
        graph, info = index_structured_file(path, output_path=cache_file, bridge_k=bridge_k)
    else:
        raise ValueError(f"Unsupported source type: {path} (pass a folder, .db/.sqlite, .jsonl/.csv, or .json.gz graph)")

    info["cached"] = False
    info["cache_file"] = str(cache_file)
    meta_file.write_text(json.dumps(info, indent=2))
    return World(graph, max_links=max_links), info


def _is_scuttle_graph(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512]
        return '"papers"' in head and '"out"' in head
    except Exception:
        return False


def _looks_like_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(16) == b"SQLite format 3\x00"
    except Exception:
        return False
