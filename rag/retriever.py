"""Retriever híbrido: combina búsqueda semántica (embeddings) y léxica (texto completo)
mediante Reciprocal Rank Fusion (RRF).

Por qué híbrido: en derecho el significado importa (semántica) pero también el término
exacto — "fuero de maternidad", "justa causa" — que un embedding puede diluir. RRF fusiona
ambos rankings sin necesidad de calibrar pesos entre puntajes de escalas distintas.
"""
from __future__ import annotations

from .config import get_config
from .db import connect, keyword_search, semantic_search
from .embeddings import embed_query

RRF_K = 60  # constante estándar de Reciprocal Rank Fusion


def hybrid_search(query: str, k: int | None = None, pool: int = 20) -> list[dict]:
    """Devuelve los k artículos más relevantes fusionando semántica + léxica.

    pool: cuántos candidatos pedir a cada buscador antes de fusionar.
    """
    cfg = get_config()
    k = k or cfg.top_k

    conn = connect()
    try:
        q_emb = embed_query(query)
        semantic = semantic_search(conn, q_emb, pool)
        lexical = keyword_search(conn, query, pool)
    finally:
        conn.close()

    fused = _reciprocal_rank_fusion([semantic, lexical])
    return fused[:k]


def _reciprocal_rank_fusion(rankings: list[list[dict]]) -> list[dict]:
    """Suma 1/(RRF_K + rank) de cada lista; ordena por puntaje combinado."""
    scores: dict[int, float] = {}
    by_id: dict[int, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            by_id[doc_id] = item

    ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    result = []
    for doc_id in ordered_ids:
        item = dict(by_id[doc_id])
        item["rrf_score"] = scores[doc_id]
        result.append(item)
    return result
