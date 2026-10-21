"""Flujo de una consulta: recuperar → decidir si hay evidencia → responder → verificar citas.

Estados posibles de una respuesta:
- ``respondida``: tiene citas válidas. Es un borrador para el equipo legal, que
  conserva el control final (``validacion_legal = "pendiente"``).
- ``abstencion``: la evidencia es débil o el generador dijo que no la encontró.
  El sistema dice "no sé" en vez de rellenar.
- ``revision_humana``: el generador respondió pero sus citas no se sostienen.
  La respuesta se rechaza y queda guardada solo para auditoría.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .answerers import NOT_FOUND
from .citations import verify_citations

ABSTAIN_MESSAGE = ("No encontré evidencia suficiente en los documentos para responder. "
                   "Deriva la consulta al equipo legal.")


@dataclass
class QueryResult:
    question: str
    status: str
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    coverage: float = 0.0
    reason: str = ""
    rejected_answer: str = ""
    validacion_legal: str = ""


def ask(question: str, retriever, answerer, k: int = 5, threshold: float = 0.35) -> QueryResult:
    hits = retriever.search(question, k=k)
    retrieved = [{"cita": h.chunk.citation, "titulo": f"{h.chunk.doc_title} — {h.chunk.section_title}",
                  "score": round(h.score, 4)} for h in hits]
    coverage = max((retriever.coverage(question, h.chunk) for h in hits), default=0.0)
    base = dict(question=question, retrieved=retrieved, coverage=round(coverage, 3))

    # 1) Umbral de "no sé": si los términos importantes de la pregunta no están
    #    en ningún pasaje recuperado, ni siquiera se llama al generador.
    if coverage < threshold:
        return QueryResult(status="abstencion", answer=ABSTAIN_MESSAGE,
                           reason=f"cobertura {coverage:.2f} bajo el umbral {threshold:.2f}", **base)

    raw = answerer.answer(question, hits).strip()
    if raw == NOT_FOUND or raw.startswith(NOT_FOUND):
        return QueryResult(status="abstencion", answer=ABSTAIN_MESSAGE,
                           reason="el generador no encontró respaldo en los pasajes", **base)

    # 2) No inventar fuentes: solo valen citas de los pasajes efectivamente recuperados.
    check = verify_citations(raw, {h.chunk.citation for h in hits})
    if not check.valid:
        return QueryResult(status="revision_humana", answer=ABSTAIN_MESSAGE, citations=check.cited,
                           reason=check.reason, rejected_answer=raw, **base)

    return QueryResult(status="respondida", answer=raw, citations=check.cited,
                       reason="citas verificadas", validacion_legal="pendiente", **base)
