"""DocuMind engine: BM25 + dense vectors, rank fusion, diversity re-ranking, LLM answers."""
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache

import numpy as np

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
STOPWORDS = set("a an the of and or to in on for is are was were be been by with as at from "
                "that this it its into than then so if not no".split())


# ----------------------------- embeddings -----------------------------
@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # ONNX runtime, no PyTorch needed
    return TextEmbedding(model_name=EMBED_MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))


def _unit(m):
    m = np.asarray(m, dtype="float32")
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def embed_passages(texts):
    return _unit(list(_model().passage_embed(texts)))


def embed_query(query):
    return _unit(list(_model().query_embed(query)))[0]


# ----------------------------- retrieval maths -----------------------------
def tok(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 written from scratch."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tf), dtype="float32")
        for w in set(query_tokens):
            if w not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    out[i] += self.idf[w] * f * (self.k1 + 1) / norm
        return out


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion: merge rankings without tuning score weights."""
    fused = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def mmr(candidates, vecs, relevance, k, diversity):
    """Maximal Marginal Relevance: pick relevant chunks that are not near-duplicates."""
    chosen, pool = [], list(candidates)
    while pool and len(chosen) < k:
        def value(i):
            redundancy = max((float(vecs[i] @ vecs[j]) for j in chosen), default=0.0)
            return (1 - diversity) * relevance[i] - diversity * redundancy
        best = max(pool, key=value)
        chosen.append(best)
        pool.remove(best)
    return chosen


# ----------------------------- library -----------------------------
class Library:
    def __init__(self):
        self.chunks, self.docs = [], {}
        self.vecs = np.zeros((0, 0), dtype="float32")
        self.bm25 = None

    def _reindex(self):
        self.bm25 = BM25([tok(c["text"]) for c in self.chunks]) if self.chunks else None

    def add(self, chunks, info, name):
        seen = {c["id"] for c in self.chunks}
        new = [c for c in chunks if c["id"] not in seen]
        self.docs[name] = info
        if not new:
            return 0
        vecs = embed_passages([c["text"] for c in new])
        self.vecs = vecs if not self.chunks else np.vstack([self.vecs, vecs])
        self.chunks = self.chunks + new
        self._reindex()
        return len(new)

    def remove(self, name):
        keep = [i for i, c in enumerate(self.chunks) if c["doc"] != name]
        self.chunks = [self.chunks[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else np.zeros((0, 0), dtype="float32")
        self.docs.pop(name, None)
        self._reindex()

    def clear(self):
        self.__init__()

    def search(self, query, k=5, mode="Hybrid", diversity=0.25, docs=None):
        ok = [i for i, c in enumerate(self.chunks) if not docs or c["doc"] in docs]
        if not ok:
            return []
        qv = embed_query(query)
        dense = self.vecs @ qv
        rankings = []
        if mode in ("Hybrid", "Semantic"):
            rankings.append(sorted(ok, key=lambda i: -dense[i]))
        if mode in ("Hybrid", "Keyword"):
            sparse = self.bm25.scores(tok(query))
            rankings.append([i for i in sorted(ok, key=lambda i: -sparse[i]) if sparse[i] > 0])
        fused = rrf(rankings)
        if not fused:
            return []
        top = max(fused.values())
        relevance = {i: v / top for i, v in fused.items()}
        pool = sorted(fused, key=lambda i: -fused[i])[: max(k * 3, 10)]
        picked = mmr(pool, self.vecs, relevance, k, diversity)
        return [{**self.chunks[i], "relevance": relevance[i], "similarity": float(dense[i])}
                for i in picked]

    def save(self, folder="vector_store"):
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(os.path.join(folder, "library.npz"), vecs=self.vecs)
        with open(os.path.join(folder, "library.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "docs": self.docs}, f)

    def load(self, folder="vector_store"):
        npz, meta = os.path.join(folder, "library.npz"), os.path.join(folder, "library.json")
        if not (os.path.exists(npz) and os.path.exists(meta)):
            return False
        self.vecs = np.load(npz)["vecs"]
        with open(meta, encoding="utf-8") as f:
            data = json.load(f)
        self.chunks, self.docs = data["chunks"], data["docs"]
        self._reindex()
        return True


# ----------------------------- language model -----------------------------
SYSTEM = (
    "You are DocuMind, a careful research assistant. Answer using ONLY the numbered "
    "excerpts provided. After each claim, cite the excerpt number in square brackets, "
    "like [1] or [2][3]. If the excerpts do not contain the answer, say so instead of "
    "guessing. Be concise and well organised."
)


def _pages(h):
    return f"p.{h['page']}" if h["page"] == h["page_end"] else f"pp.{h['page']}-{h['page_end']}"


def _context(hits):
    return "\n\n".join(f"[{n}] ({h['doc']}, {_pages(h)})\n{h['text']}" for n, h in enumerate(hits, 1))


def _llm_stream(system, messages, provider, api_key=None, max_tokens=1024):
    if provider == "Claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        with client.messages.stream(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                                    max_tokens=max_tokens, system=system, messages=messages) as s:
            yield from s.text_stream
    elif provider == "Ollama":
        import requests
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat"
        body = {"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": True,
                "messages": [{"role": "system", "content": system}] + messages}
        with requests.post(url, json=body, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line).get("message", {}).get("content", "")


def _extractive(question, hits):
    q = set(tok(question))
    scored = [(len(q & set(tok(s))), n, s)
              for n, h in enumerate(hits, 1) for s in re.split(r"(?<=[.!?])\s+", h["text"])]
    scored.sort(key=lambda x: -x[0])
    yield "*No language model selected, so these are the most relevant sentences:*\n\n"
    for _, n, s in scored[:4]:
        yield f"- {s} [{n}]\n"


def stream_answer(question, hits, provider="Claude", api_key=None, history=None):
    if not hits:
        yield "I couldn't find anything relevant in the selected documents."
        return
    if provider == "Extractive (no AI)":
        yield from _extractive(question, hits)
        return
    msgs = (history or [])[-6:] + [
        {"role": "user", "content": f"Excerpts:\n\n{_context(hits)}\n\nQuestion: {question}"}]
    yield from _llm_stream(SYSTEM, msgs, provider, api_key)


def summarize(lib, doc, provider="Claude", api_key=None):
    """Brief overview of one document plus suggested questions."""
    chunks = [c for c in lib.chunks if c["doc"] == doc]
    step = max(1, len(chunks) // 10)
    sample = "\n\n".join(c["text"] for c in chunks[::step][:10])
    if provider == "Extractive (no AI)":
        return "Pick Claude or Ollama for summaries. Opening text:\n\n" + sample[:600]
    prompt = (f"Document: {doc}\n\nSampled passages:\n{sample}\n\n"
              "Write a 4-bullet overview of what this document is about, then a short "
              "'Try asking' list with 3 specific questions a reader could ask.")
    return "".join(_llm_stream("You summarise documents clearly and briefly.",
                               [{"role": "user", "content": prompt}], provider, api_key, 600))
