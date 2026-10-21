"""CLI.

    python -m semsearch ask "¿Puedo volar en clase ejecutiva?"
    python -m semsearch eval --answerer extractive
    python -m semsearch sweep          # barrido del umbral de abstención
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .answerers import get_answerer
from .corpus import load_corpus
from .evaluate import evaluate
from .pipeline import ask
from .retriever import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Búsqueda semántica con citas obligatorias")
    parser.add_argument("command", choices=["ask", "eval", "sweep"])
    parser.add_argument("question", nargs="?", help="pregunta (solo para 'ask')")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--questions", default="data/eval_questions.jsonl")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    parser.add_argument("--answerer", default="extractive", choices=["extractive", "openai", "anthropic"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="cobertura mínima de la pregunta para responder (0 a 1)")
    parser.add_argument("--out", default="output/eval_results.jsonl")
    args = parser.parse_args()

    retriever = get_retriever(args.retriever, load_corpus(args.corpus))
    answerer = get_answerer(args.answerer, retriever)

    if args.command == "ask":
        if not args.question:
            parser.error("falta la pregunta")
        result = ask(args.question, retriever, answerer, k=args.k, threshold=args.threshold)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    if args.command == "sweep":
        # Muestra el costo de cada umbral: más alto = más abstenciones correctas,
        # pero también más preguntas con respuesta que quedan sin contestar.
        print("umbral | citadas OK % | cita incorrecta | abstenciones indebidas | abstención correcta %")
        for threshold in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
            res = [ask(q["question"], retriever, answerer, k=args.k, threshold=threshold) for q in questions]
            m = evaluate(res, questions)
            print(f"{threshold:.2f} | {m['respuestas_citadas_correctamente_%']} | "
                  f"{m['respondidas_con_cita_incorrecta']} | {m['abstenciones_en_preguntas_con_respuesta']} | "
                  f"{m['abstencion_correcta_sin_respuesta_%']}")
        return

    results = [ask(q["question"], retriever, answerer, k=args.k, threshold=args.threshold) for q in questions]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps({"id": q["id"], "expected": q["expected"], **asdict(r)},
                                        ensure_ascii=False) for q, r in zip(questions, results)) + "\n",
                   encoding="utf-8")

    report = {"retriever": retriever.name, "answerer": answerer.name, "k": args.k,
              "threshold": args.threshold, **evaluate(results, questions)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
