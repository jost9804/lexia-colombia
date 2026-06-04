"""Evaluación del sistema RAG. Mide dos cosas:

1. RETRIEVAL — recall@k: ¿el artículo correcto aparece entre los k recuperados?
   Es la métrica más importante: si no recuperas el artículo, no hay forma de responder bien.

2. GENERACIÓN — faithfulness (LLM-as-judge): ¿la respuesta se apoya solo en el contexto,
   sin inventar? Usamos a Claude como juez con una rúbrica estricta.

Uso:
    python -m eval.evaluate --k 6
"""
from __future__ import annotations

import argparse
import json
import os
import re

from rag.generator import answer, llm_complete
from rag.retriever import hybrid_search

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.jsonl")

JUDGE_PROMPT = """\
Eres un evaluador estricto. Te doy un CONTEXTO (artículos legales) y una RESPUESTA.
Decide si la RESPUESTA está completamente respaldada por el CONTEXTO (faithfulness).

Responde SOLO con un número:
1 = toda afirmación de la respuesta se apoya en el contexto (o admite no tener respaldo).
0 = la respuesta contiene afirmaciones no respaldadas por el contexto (alucinación).

CONTEXTO:
{context}

RESPUESTA:
{answer}

Tu veredicto (solo 1 o 0):"""


def load_dataset() -> list[dict]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_retrieval(dataset: list[dict], k: int) -> float:
    hits = 0
    for item in dataset:
        retrieved = {c["article_no"] for c in hybrid_search(item["question"], k=k)}
        if set(item["expected_articles"]) & retrieved:
            hits += 1
            mark = "✓"
        else:
            mark = "✗"
        print(f"  {mark} recall  | {item['question'][:60]}")
    recall = hits / len(dataset)
    print(f"\nRecall@{k}: {recall:.2%}  ({hits}/{len(dataset)})")
    return recall


def eval_faithfulness(dataset: list[dict], k: int) -> float:
    faithful = 0
    for item in dataset:
        chunks = hybrid_search(item["question"], k=k)
        context = "\n\n".join(f"[Art. {c['article_no']}] {c['content']}" for c in chunks)
        result = answer(item["question"], k=k)
        score_text = llm_complete(
            JUDGE_PROMPT.format(context=context, answer=result["answer"])
        )
        score = 1 if re.search(r"\b1\b", score_text) else 0
        faithful += score
        print(f"  {'✓' if score else '✗'} faithful | {item['question'][:60]}")
    rate = faithful / len(dataset)
    print(f"\nFaithfulness: {rate:.2%}  ({faithful}/{len(dataset)})")
    return rate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--skip-faithfulness", action="store_true",
                   help="Solo evalúa retrieval (no consume tokens de generación).")
    args = p.parse_args()

    dataset = load_dataset()
    print(f"Evaluando {len(dataset)} preguntas…\n")
    print("=== RETRIEVAL ===")
    recall = eval_retrieval(dataset, args.k)

    if not args.skip_faithfulness:
        print("\n=== FAITHFULNESS (LLM-as-judge) ===")
        faith = eval_faithfulness(dataset, args.k)
        print(f"\nRESUMEN  ·  Recall@{args.k}: {recall:.2%}  ·  Faithfulness: {faith:.2%}")


if __name__ == "__main__":
    main()
