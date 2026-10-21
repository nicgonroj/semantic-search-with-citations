"""Ingesta del corpus: un archivo Markdown por documento, un fragmento por sección.

Cada documento tiene un frontmatter con ``doc_id`` y ``titulo``, y secciones
``## N. Título``. Trocear por sección (y no por cantidad de caracteres) hace
que cada fragmento sea citable tal como lo citaría un abogado: ``[POL-005 §2]``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.M)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str          # número de sección como texto: "2"
    doc_title: str
    section_title: str
    text: str

    @property
    def citation(self) -> str:
        """Identificador citable del fragmento, p. ej. ``POL-005 §2``."""
        return f"{self.doc_id} §{self.section}"

    @property
    def indexed_text(self) -> str:
        # El título del documento y de la sección se indexan junto al cuerpo:
        # muchas preguntas nombran el tema ("viáticos") más que el contenido.
        return f"{self.doc_title}. {self.section_title}. {self.text}"


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def parse_document(raw: str) -> list[Chunk]:
    meta, body = _parse_frontmatter(raw)
    doc_id, title = meta["doc_id"], meta.get("titulo", meta["doc_id"])
    matches = list(SECTION_RE.finditer(body))
    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = " ".join(body[match.end():end].split())
        if text:
            chunks.append(Chunk(doc_id, match.group(1), title, match.group(2).strip(), text))
    return chunks


def load_corpus(folder: str | Path) -> list[Chunk]:
    """Lee todos los ``.md`` de la carpeta en orden estable."""
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        chunks.extend(parse_document(path.read_text(encoding="utf-8")))
    if not chunks:
        raise FileNotFoundError(f"no hay documentos .md en {folder}")
    return chunks
