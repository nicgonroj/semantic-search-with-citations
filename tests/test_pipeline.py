import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semsearch import answerers  # noqa: E402
from semsearch.answerers import NOT_FOUND, ExtractiveAnswerer  # noqa: E402
from semsearch.citations import extract_citations, verify_citations  # noqa: E402
from semsearch.corpus import load_corpus, parse_document  # noqa: E402
from semsearch.evaluate import evaluate  # noqa: E402
from semsearch.pipeline import ask  # noqa: E402
from semsearch.retriever import BM25Retriever, HybridRetriever  # noqa: E402
from semsearch.text import tokenize  # noqa: E402

CORPUS = load_corpus(ROOT / "data" / "corpus")

SAMPLE = """---
doc_id: TST-001
titulo: Documento de prueba
---
# Documento de prueba

## 1. Primera sección

Texto uno.

## 2. Segunda sección

Texto dos, en
varias líneas.
"""


class FixedAnswerer:
    """Generador falso que devuelve siempre el mismo texto (simula un LLM)."""

    name = "fijo"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def answer(self, question, hits):
        self.calls += 1
        return self.text


class TestCorpus(unittest.TestCase):
    def test_chunk_per_section(self):
        chunks = parse_document(SAMPLE)
        self.assertEqual([c.citation for c in chunks], ["TST-001 §1", "TST-001 §2"])
        self.assertEqual(chunks[1].text, "Texto dos, en varias líneas.")

    def test_corpus_loaded(self):
        self.assertEqual(len({c.doc_id for c in CORPUS}), 40)
        self.assertEqual(len({c.citation for c in CORPUS}), len(CORPUS))


class TestText(unittest.TestCase):
    def test_accents_and_plurals_match(self):
        self.assertEqual(tokenize("Contratos"), tokenize("contrato"))
        self.assertEqual(tokenize("viático"), tokenize("viaticos"))
        self.assertNotIn("de", tokenize("plazo de pago"))


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(CORPUS)

    def test_top_hit(self):
        hits = self.retriever.search("¿Cuántos caracteres debe tener una contraseña?", k=3)
        self.assertEqual(hits[0].chunk.citation, "POL-004 §2")

    def test_coverage_low_for_unrelated_question(self):
        query = "¿Se pueden traer mascotas a la oficina?"
        hits = self.retriever.search(query, k=5)
        coverage = max((self.retriever.coverage(query, h.chunk) for h in hits), default=0.0)
        self.assertLess(coverage, 0.35)


class BagOfWordsEmbedder:
    """Embeddings falsos (conteo de términos) para probar la fusión sin red."""

    def __init__(self):
        self.vocab = sorted({t for c in CORPUS for t in tokenize(c.indexed_text)})

    def embed(self, texts):
        vectors = []
        for text in texts:
            terms = tokenize(text)
            vectors.append([float(terms.count(t)) for t in self.vocab])
        return vectors


class TestHybridRetriever(unittest.TestCase):
    def test_fusion_keeps_obvious_hit_first(self):
        retriever = HybridRetriever(CORPUS, embedder=BagOfWordsEmbedder())
        hits = retriever.search("¿Cuántos caracteres debe tener una contraseña?", k=3)
        self.assertEqual(hits[0].chunk.citation, "POL-004 §2")
        self.assertEqual(len(hits), 3)


class TestCitations(unittest.TestCase):
    def test_extract_multiple_in_one_bracket(self):
        self.assertEqual(extract_citations("Algo [POL-001 §2; POL-003 §1]. Otra [POL-001 §2]"),
                         ["POL-001 §2", "POL-003 §1"])

    def test_rejects_missing_citation(self):
        self.assertFalse(verify_citations("Se paga en 30 días.", {"CON-001 §2"}).valid)

    def test_rejects_invented_source(self):
        check = verify_citations("Se paga en 30 días [CON-099 §7].", {"CON-001 §2"})
        self.assertFalse(check.valid)
        self.assertEqual(check.invalid, ["CON-099 §7"])

    def test_rejects_uncited_paragraph(self):
        text = "Se paga en 30 días [CON-001 §2].\n\nY además hay un descuento por pronto pago."
        self.assertFalse(verify_citations(text, {"CON-001 §2"}).valid)

    def test_accepts_valid(self):
        self.assertTrue(verify_citations("Se paga en 30 días [CON-001 §2].", {"CON-001 §2"}).valid)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(CORPUS)
        self.question = "¿Puedo volar en clase ejecutiva en un viaje de trabajo?"

    def test_extractive_answer_is_cited(self):
        result = ask(self.question, self.retriever, ExtractiveAnswerer(self.retriever.term_idf))
        self.assertEqual(result.status, "respondida")
        self.assertIn("POL-005 §2", result.citations)
        self.assertEqual(result.validacion_legal, "pendiente")

    def test_invented_citation_goes_to_human_review(self):
        fake = FixedAnswerer("La clase ejecutiva está siempre permitida [POL-099 §1].")
        result = ask(self.question, self.retriever, fake)
        self.assertEqual(result.status, "revision_humana")
        self.assertIn("POL-099 §1", result.rejected_answer)
        self.assertNotIn("POL-099", result.answer)

    def test_existing_but_not_retrieved_citation_is_rejected(self):
        # PRO-010 §3 existe en el corpus, pero no fue recuperado para esta pregunta.
        fake = FixedAnswerer("Se recupera en 8 horas [PRO-010 §3].")
        result = ask(self.question, self.retriever, fake)
        self.assertEqual(result.status, "revision_humana")

    def test_model_says_not_found(self):
        result = ask(self.question, self.retriever, FixedAnswerer(NOT_FOUND))
        self.assertEqual(result.status, "abstencion")

    def test_weak_evidence_skips_generator(self):
        fake = FixedAnswerer("No debería llamarse [POL-001 §1].")
        result = ask("¿Se pueden traer mascotas a la oficina?", self.retriever, fake)
        self.assertEqual(result.status, "abstencion")
        self.assertEqual(fake.calls, 0)


class TestEvaluate(unittest.TestCase):
    def test_metrics(self):
        retriever = BM25Retriever(CORPUS)
        answerer = ExtractiveAnswerer(retriever.term_idf)
        questions = [
            {"question": "¿Cuántos caracteres debe tener una contraseña?", "expected": ["POL-004 §2"]},
            {"question": "¿Se pueden traer mascotas a la oficina?", "expected": []},
        ]
        results = [ask(q["question"], retriever, answerer) for q in questions]
        report = evaluate(results, questions)
        self.assertEqual(report["recall@1"], 100.0)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["respuestas_citadas_correctamente_%"], 100.0)
        self.assertEqual(report["abstencion_correcta_sin_respuesta_%"], 100.0)


class TestAnthropicAnswerer(unittest.TestCase):
    def test_default_model_and_parsing(self):
        captured = {}

        def fake_post(url, headers, payload, timeout=60):
            captured.update(url=url, payload=payload)
            return {"content": [{"type": "text", "text": "Respuesta [POL-005 §2]"}]}

        env = {"ANTHROPIC_API_KEY": "clave-de-prueba"}
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(answerers, "_post_json", fake_post):
            os.environ.pop("ANTHROPIC_MODEL", None)
            text = answerers.AnthropicAnswerer().answer("pregunta", [])
        self.assertEqual(captured["payload"]["model"], "claude-sonnet-5-5")
        self.assertTrue(captured["url"].endswith("/v1/messages"))
        self.assertEqual(text, "Respuesta [POL-005 §2]")


if __name__ == "__main__":
    unittest.main()
