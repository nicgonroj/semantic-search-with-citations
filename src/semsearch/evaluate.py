"""Métricas de recuperación y de respuesta sobre el set de evaluación.

- ``recall@k``: en preguntas con respuesta, ¿la sección correcta está entre los k primeros?
- ``MRR``: promedio de 1/rango de la primera sección correcta (0 si no aparece).
- ``respuestas citadas correctamente``: preguntas con respuesta en que el sistema
  respondió y citó una de las secciones esperadas.
- ``abstención correcta``: preguntas sin respuesta en el corpus en que el sistema
  no respondió (se abstuvo o derivó a revisión humana).
"""
from __future__ import annotations

from .pipeline import QueryResult


def _pct(num: int, den: int) -> float:
    return round(100 * num / den, 1) if den else 0.0


def evaluate(results: list[QueryResult], questions: list[dict], ks=(1, 3, 5)) -> dict:
    answerable = [(r, q) for r, q in zip(results, questions) if q["expected"]]
    unanswerable = [(r, q) for r, q in zip(results, questions) if not q["expected"]]

    def first_rank(result: QueryResult, expected: set[str]) -> int | None:
        for rank, item in enumerate(result.retrieved, start=1):
            if item["cita"] in expected:
                return rank
        return None

    ranks = [first_rank(r, set(q["expected"])) for r, q in answerable]
    recall = {f"recall@{k}": _pct(sum(1 for x in ranks if x and x <= k), len(ranks)) for k in ks}
    mrr = round(sum(1 / x for x in ranks if x) / len(ranks), 3) if ranks else 0.0

    answered = [(r, q) for r, q in answerable if r.status == "respondida"]
    correct = [r for r, q in answered if set(r.citations) & set(q["expected"])]
    abstained_ok = [r for r, _ in unanswerable if r.status != "respondida"]
    rejected = [r for r in results if r.status == "revision_humana"]

    return {
        "preguntas": len(results),
        "con_respuesta": len(answerable),
        "sin_respuesta": len(unanswerable),
        **recall,
        "mrr": mrr,
        "respuestas_citadas_correctamente_%": _pct(len(correct), len(answerable)),
        "respondidas_con_cita_incorrecta": len(answered) - len(correct),
        "abstenciones_en_preguntas_con_respuesta": len(answerable) - len(answered),
        "abstencion_correcta_sin_respuesta_%": _pct(len(abstained_ok), len(unanswerable)),
        "rechazadas_por_citas_invalidas": len(rejected),
    }
