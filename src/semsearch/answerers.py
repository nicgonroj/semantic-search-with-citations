"""Generadores de respuesta a partir de los pasajes recuperados.

- ``ExtractiveAnswerer``: elige las oraciones de los pasajes que mejor cubren la
  pregunta y las devuelve textuales, cada una con su cita. Sin red ni claves.
- ``OpenAIAnswerer`` y ``AnthropicAnswerer``: redactan una respuesta en español
  con citas obligatorias. Solo se usan si existe la variable de entorno con la
  clave. Su salida pasa por la misma verificación de citas que la extractiva.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .retriever import Hit
from .text import split_sentences, tokenize

NOT_FOUND = "NO_ENCONTRADO"

SYSTEM_PROMPT = (
    "Eres un asistente de búsqueda para el equipo legal de una empresa. Respondes en español, "
    "de forma breve y precisa, usando EXCLUSIVAMENTE los pasajes entregados. Cada párrafo de tu "
    "respuesta debe terminar con la cita del pasaje que lo respalda, con el formato exacto "
    "[DOC-ID §N], por ejemplo [POL-005 §2]. No cites pasajes que no te fueron entregados ni "
    "inventes documentos, plazos o montos. Si los pasajes no permiten responder, responde "
    f"exactamente {NOT_FOUND} y nada más."
)


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.citation}] {h.chunk.doc_title} — {h.chunk.section_title}\n{h.chunk.text}" for h in hits
    )


class ExtractiveAnswerer:
    """Respuesta textual: nunca dice algo que no esté escrito en un pasaje."""

    name = "extractive"

    def __init__(self, idf_of, max_sentences: int = 2, top_passages: int = 3):
        self.idf_of = idf_of            # función término → IDF, la entrega el recuperador
        self.max_sentences = max_sentences
        self.top_passages = top_passages

    def _sentence_score(self, query_terms: set[str], sentence: str, rank: int) -> float:
        terms = set(tokenize(sentence))
        total = sum(self.idf_of(t) for t in query_terms) or 1.0
        overlap = sum(self.idf_of(t) for t in query_terms if t in terms) / total
        return overlap - 0.05 * (rank - 1)   # a igual cobertura, manda el pasaje mejor rankeado

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return NOT_FOUND
        query_terms = set(tokenize(question))
        candidates = []
        for hit in hits[: self.top_passages]:
            for sentence in split_sentences(hit.chunk.text):
                score = self._sentence_score(query_terms, sentence, hit.rank)
                candidates.append((score, sentence, hit.chunk.citation))
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][0]
        if best <= 0:
            return NOT_FOUND
        chosen = [c for c in candidates if c[0] >= 0.6 * best][: self.max_sentences]
        return "\n\n".join(f"{sentence} [{citation}]" for _, sentence, citation in chosen)


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _user_prompt(question: str, hits: list[Hit]) -> str:
    return f"Pasajes:\n\n{format_passages(hits)}\n\nPregunta: {question}"


class OpenAIAnswerer:
    name = "openai"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, question: str, hits: list[Hit]) -> str:
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": _user_prompt(question, hits)}]}
        data = _post_json("https://api.openai.com/v1/chat/completions",
                          {"Authorization": f"Bearer {self.api_key}"}, payload)
        return data["choices"][0]["message"]["content"].strip()


class AnthropicAnswerer:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def answer(self, question: str, hits: list[Hit]) -> str:
        payload = {"model": self.model, "max_tokens": 600, "system": SYSTEM_PROMPT,
                   "messages": [{"role": "user", "content": _user_prompt(question, hits)}]}
        data = _post_json("https://api.anthropic.com/v1/messages",
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, payload)
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()


def get_answerer(name: str, retriever):
    if name == "extractive":
        return ExtractiveAnswerer(retriever.term_idf)
    if name == "openai":
        return OpenAIAnswerer()
    if name == "anthropic":
        return AnthropicAnswerer()
    raise ValueError(f"generador desconocido: {name}")
