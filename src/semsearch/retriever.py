"""Recuperadores de fragmentos.

- ``BM25Retriever``: BM25 implementado con biblioteca estándar. Sin red ni
  claves; es el modo por defecto y la base reproducible de las métricas.
- ``HybridRetriever``: combina BM25 con embeddings de OpenAI mediante Reciprocal
  Rank Fusion. Solo se usa si existe ``OPENAI_API_KEY``.

Además de ordenar, el recuperador mide cuánta evidencia hay: ``coverage`` es la
fracción del peso IDF de la pregunta que aparece en el fragmento. Una pregunta
cuyos términos importantes no están en ningún documento tiene cobertura baja y
el sistema se abstiene en vez de responder con lo primero que encuentre.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .corpus import Chunk
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.indexed_text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Un término que no aparece en el corpus pesa como el más raro posible.
        self.unknown_idf = math.log(1 + (n + 0.5) / 0.5)

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, self.unknown_idf)

    def _score(self, terms: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        score = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        scored = sorted(((self._score(terms, i), i) for i in range(len(self.chunks))), reverse=True)
        return [Hit(self.chunks[i], s, r) for r, (s, i) in enumerate(scored[:k], start=1) if s > 0]

    def coverage(self, query: str, chunk: Chunk) -> float:
        """Fracción del peso IDF de la pregunta presente en el fragmento (0 a 1)."""
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        present = set(tokenize(chunk.indexed_text))
        total = sum(self.term_idf(t) for t in terms)
        return sum(self.term_idf(t) for t in terms if t in present) / total


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OpenAIEmbedder:
    """Embeddings vía API de OpenAI, con caché en disco para no pagar dos veces."""

    def __init__(self, model: str | None = None, cache_path: str = "output/embeddings_cache.json"):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if self._key(t) not in self.cache]
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            data = _post_json("https://api.openai.com/v1/embeddings",
                              {"Authorization": f"Bearer {self.api_key}"},
                              {"model": self.model, "input": batch})
            for text, item in zip(batch, data["data"]):
                self.cache[self._key(text)] = item["embedding"]
        if missing:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


class HybridRetriever(BM25Retriever):
    """BM25 + embeddings, fusionados por RRF. La cobertura sigue siendo léxica."""

    name = "hybrid"

    def __init__(self, chunks: list[Chunk], embedder=None, rrf_k: int = 60):
        super().__init__(chunks)
        self.embedder = embedder or OpenAIEmbedder()
        self.rrf_k = rrf_k
        self.vectors = self.embedder.embed([c.indexed_text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        lexical = super().search(query, k=len(self.chunks))
        qvec = self.embedder.embed([query])[0]
        dense = sorted(range(len(self.chunks)), key=lambda i: _cosine(qvec, self.vectors[i]), reverse=True)
        fused: dict[int, float] = {}
        index = {id(c): i for i, c in enumerate(self.chunks)}
        for rank, hit in enumerate(lexical, start=1):
            fused[index[id(hit.chunk)]] = fused.get(index[id(hit.chunk)], 0.0) + 1 / (self.rrf_k + rank)
        for rank, i in enumerate(dense, start=1):
            fused[i] = fused.get(i, 0.0) + 1 / (self.rrf_k + rank)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(best, start=1)]


def get_retriever(name: str, chunks: list[Chunk]):
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "hybrid":
        return HybridRetriever(chunks)
    raise ValueError(f"recuperador desconocido: {name}")
