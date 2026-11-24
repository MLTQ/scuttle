"""ScuttleWorker: Single-pass 1.7B decision model with BF16 / INT8 / 4-bit NF4 support and HF Hub auto-loading."""
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from scuttle.world import stable_hash

DEFAULT_MODEL_ID = os.environ.get("SCUTTLE_MODEL", "Multiqueue/scuttle-1.7b")
THRESHOLDS = (0, 1, 2, 3, 4, 6)
ABEL_W = (0.50, 0.22, 0.12, 0.07, 0.05, 0.04)

SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)
MENU_TASK = (
    "First say whether the current paper is the paper the query describes. Then, for each numbered "
    "link, say whether following it leads toward that paper."
)
ABSTRACT_CHARS = 700
MAX_QUERY = 96
CONTEXT_CHARS, SNIPPET_SENTENCES, SNIPPET_CHARS, SHORT_TITLE = 300, 2, 300, 60
HOLE = "\x00DOC\x00"
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def clip(text, n):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def snippet(abstract, max_sentences=SNIPPET_SENTENCES, max_chars=SNIPPET_CHARS):
    return clip(" ".join(SENTENCE.split(abstract)[:max_sentences]), max_chars)


def shown_order(menu, key):
    return sorted(range(len(menu)), key=lambda i: stable_hash("order", key, menu[i][0]))


def menu_doc(world, page, menu, order):
    p = world.papers[page]
    noun = getattr(world, "doc_noun", "document")
    n_links = len(order)
    if n_links <= 28:
        ctx_chars, sn_sents, sn_chars = CONTEXT_CHARS, SNIPPET_SENTENCES, SNIPPET_CHARS
    elif n_links <= 60:
        ctx_chars, sn_sents, sn_chars = 180, 1, 160
    else:
        ctx_chars, sn_sents, sn_chars = 130, 1, 110

    out_phrase = f"The current {noun} cites it" if noun == "paper" else f"The current {noun} links to it"
    in_phrase = f"It cites the current {noun}" if noun == "paper" else f"It links to the current {noun}"

    lines = [
        f"Current {noun}: {p.get('title') or '(untitled)'}",
        f"Abstract: {clip(p.get('abstract') or '', ABSTRACT_CHARS) or '(none)'}",
        "",
        "Links:" if order else "Links: (none)",
    ]
    for n, i in enumerate(order, 1):
        dest, kind, context = menu[i]
        how = out_phrase if kind == "cites" else in_phrase
        cited = f'"{clip(context, ctx_chars)}"' if context else "(no context sentence)"
        dp = world.papers[dest]
        lines += [
            f"[{n}] {how}: {cited}",
            f"Destination: {dp.get('title') or '(untitled)'}",
            f"Abstract: {snippet(dp.get('abstract') or '', sn_sents, sn_chars) or '(none)'}",
        ]
    return "\n".join(lines)


def frontier_value(link_probs, weights=ABEL_W):
    """Compose calibrated cumulative probabilities P(d <= T_k) into a scalar frontier priority in [0, 1]."""
    w = link_probs.new_tensor(weights)
    return (link_probs * w).sum(dim=-1)


class ValueHeads(nn.Module):
    def __init__(self, hidden_size, n_bins=len(THRESHOLDS)):
        super().__init__()
        self.page_head = nn.Linear(hidden_size, 1)
        self.link_head = nn.Linear(hidden_size, n_bins)


def _resolve_heads_file(model_id_or_path):
    p = Path(model_id_or_path)
    if p.exists():
        for name in ("scuttle_heads.pt", "g4n_heads.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"No scuttle_heads.pt or g4n_heads.pt found in {p}")
    for name in ("scuttle_heads.pt", "g4n_heads.pt"):
        try:
            return Path(hf_hub_download(repo_id=str(model_id_or_path), filename=name))
        except Exception:
            continue
    raise FileNotFoundError(f"Could not download scuttle_heads.pt from Hugging Face repo {model_id_or_path}")


class ScuttleWorker:
    """Single-pass Scuttle decision worker supporting BF16, 8-bit, 4-bit NF4, and CPU execution."""

    def __init__(self, model_id_or_path=DEFAULT_MODEL_ID, device="auto", quant="none"):
        assert quant in ("none", "bf16", "8bit", "4bit"), f"Unsupported quant={quant}"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.quant = quant
        self.model_id = str(model_id_or_path)
        self.tokens = 0
        local = Path(self.model_id).exists()

        self.tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local, trust_remote_code=True)
        load_kw = dict(local_files_only=local, trust_remote_code=True)

        if device == "cpu" or quant in ("none", "bf16"):
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self.lm = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, **load_kw).to(device)
        else:
            from transformers import BitsAndBytesConfig
            if quant == "8bit":
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
            else:
                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            dev_idx = int(device.split(":")[1]) if ":" in device else 0
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=qcfg, device_map={"": dev_idx}, **load_kw
            )

        self.lm.eval()
        self.lm.config.use_cache = False

        heads_path = _resolve_heads_file(self.model_id)
        self.heads = ValueHeads(self.lm.config.hidden_size).to(device=self.device, dtype=torch.float32)
        self.heads.load_state_dict(torch.load(heads_path, map_location=self.device, weights_only=True))
        self.heads.eval()

        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self._head = {}
        _, tail = self._template(MENU_TASK, "QUERY").split(HOLE)
        self.tail = self.tok.encode(tail, add_special_tokens=False)

    def _template(self, task, query):
        user = f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {HOLE}"
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def head(self, query):
        if query not in self._head:
            q = self.tok.decode(self.tok.encode(query, add_special_tokens=False)[:MAX_QUERY])
            self._head[query] = self.tok.encode(self._template(MENU_TASK, q).split(HOLE)[0], add_special_tokens=False)
        return self._head[query]

    def menu_row(self, world, query, page, hidden=(), key="eval", skip=(), menu_override=None):
        base_menu = menu_override if menu_override is not None else world.menu(page, hidden)
        menu = [m for m in base_menu if m[0] not in skip] if skip else list(base_menu)
        order = shown_order(menu, f"{key}:{page}")
        doc = menu_doc(world, page, menu, order)
        ids = self.head(query) + self.tok.encode(doc, add_special_tokens=False) + self.tail
        at = []
        noun = getattr(world, "doc_noun", "document")
        lines = [f"\nCurrent {noun}:"] + [
            f"\n[{n}] {clip(world.papers[menu[i][0]].get('title') or '(untitled)', SHORT_TITLE)}:"
            for n, i in enumerate(order, 1)
        ]
        for line in lines:
            ids = ids + self.tok.encode(line, add_special_tokens=False)
            at.append(len(ids) - 1)
        return dict(ids=ids, at=at, order=order, dests=[m[0] for m in menu])

    @torch.inference_mode()
    def menu_outputs(self, rows):
        n = max(len(r["ids"]) for r in rows)
        ids = torch.full((len(rows), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            L = len(r["ids"])
            ids[i, :L] = torch.tensor(r["ids"], dtype=torch.long)
            mask[i, :L] = 1
        self.tokens += int(mask.sum())

        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else _nullctx()
        with ctx:
            hidden = self.lm.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            ).last_hidden_state

        b = torch.tensor([i for i, r in enumerate(rows) for _ in r["at"]], device=self.device, dtype=torch.long)
        t = torch.tensor([a for r in rows for a in r["at"]], device=self.device, dtype=torch.long)
        h_all = hidden[b, t].float()

        out = []
        splits = [len(r["at"]) for r in rows]
        for r, hr in zip(rows, h_all.split(splits)):
            page_z = self.heads.page_head(hr[0:1]).squeeze(-1).squeeze(0)
            if len(r["order"]) == 0:
                link_z = hr.new_zeros((0, len(THRESHOLDS)))
            else:
                shown_z = self.heads.link_head(hr[1:])
                shown_pos = {menu_idx: pos for pos, menu_idx in enumerate(r["order"])}
                inv = torch.tensor([shown_pos[i] for i in range(len(r["order"]))], device=hr.device, dtype=torch.long)
                link_z = shown_z[inv]
            out.append((page_z, link_z))
        return out

    def score_rows(self, rows, budget=32768):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
        out, chunk = [None] * len(rows), []
        for i in order + [None]:
            if chunk and (i is None or (len(chunk) + 1) * len(rows[i]["ids"]) > budget):
                for j, res in zip(chunk, self.menu_outputs([rows[j] for j in chunk])):
                    out[j] = res
                chunk = []
            if i is not None:
                chunk.append(i)
        return out


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False
