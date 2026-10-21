"""Verificación de citas: la regla "no inventar fuentes".

Una respuesta solo se acepta si:
1. trae al menos una cita con el formato ``[DOC-ID §N]``;
2. cada cita corresponde a un fragmento que el recuperador entregó para esa
   pregunta (no basta con que el documento exista: tiene que haber sido leído);
3. cada párrafo con contenido lleva al menos una cita.

Si algo falla, la respuesta se rechaza completa y pasa a revisión humana.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

BRACKET_RE = re.compile(r"\[([^\]]+)\]")
CITATION_RE = re.compile(r"([A-Z]{2,5}-\d{3})\s*§\s*(\d+)")


def extract_citations(text: str) -> list[str]:
    """Devuelve las citas en orden de aparición, sin duplicados.

    Acepta varias citas en un mismo corchete: ``[POL-001 §2; POL-003 §1]``.
    """
    found: list[str] = []
    for bracket in BRACKET_RE.findall(text):
        for doc_id, section in CITATION_RE.findall(bracket):
            citation = f"{doc_id} §{section}"
            if citation not in found:
                found.append(citation)
    return found


@dataclass
class CitationCheck:
    valid: bool
    cited: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    reason: str = ""


def verify_citations(answer: str, allowed: set[str]) -> CitationCheck:
    cited = extract_citations(answer)
    if not cited:
        return CitationCheck(False, reason="la respuesta no trae citas")
    invalid = [c for c in cited if c not in allowed]
    if invalid:
        return CitationCheck(False, cited, invalid,
                             f"cita fuentes que no están entre los pasajes recuperados: {', '.join(invalid)}")
    paragraphs = [p for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
    uncited = [p for p in paragraphs if not extract_citations(p)]
    if uncited:
        return CitationCheck(False, cited, reason=f"{len(uncited)} párrafo(s) sin cita")
    return CitationCheck(True, cited)
