"""Acceso a PostgreSQL + pgvector: esquema, inserción y búsquedas semántica/léxica."""
from __future__ import annotations

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import get_config


def _to_vector(embedding) -> np.ndarray:
    """pgvector + psycopg adapta de forma fiable los arreglos numpy float32."""
    return np.asarray(embedding, dtype=np.float32)


def connect() -> psycopg.Connection:
    cfg = get_config()
    # prepare_threshold=None desactiva prepared statements (compatibilidad con el pooler).
    conn = psycopg.connect(cfg.database_url, autocommit=True, prepare_threshold=None)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Crea la tabla e índices con la dimensión configurada (idempotente)."""
    cfg = get_config()
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS articles (
            id          BIGSERIAL PRIMARY KEY,
            code        TEXT    NOT NULL,
            article_no  INTEGER,
            title       TEXT,
            section     TEXT,
            content     TEXT    NOT NULL,
            embedding   vector({cfg.embed_dim}),
            tsv         tsvector
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS articles_embedding_idx "
        "ON articles USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS articles_tsv_idx ON articles USING gin (tsv)"
    )


def clear_code(conn: psycopg.Connection, code: str) -> None:
    """Borra los artículos de un código para reingestar limpio."""
    conn.execute("DELETE FROM articles WHERE code = %s", (code,))


def insert_articles(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Inserta artículos con su embedding y genera el tsvector en español."""
    rows = [{**r, "embedding": _to_vector(r["embedding"])} for r in rows]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO articles (code, article_no, title, section, content, embedding, tsv)
            VALUES (%(code)s, %(article_no)s, %(title)s, %(section)s, %(content)s,
                    %(embedding)s, to_tsvector('spanish', %(content)s))
            """,
            rows,
        )


def semantic_search(conn: psycopg.Connection, query_embedding, k: int) -> list[dict]:
    """Top-k por distancia coseno (operador <=> de pgvector)."""
    query_embedding = _to_vector(query_embedding)
    rows = conn.execute(
        """
        SELECT id, code, article_no, title, section, content,
               1 - (embedding <=> %s) AS score
        FROM articles
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_embedding, query_embedding, k),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def keyword_search(conn: psycopg.Connection, query: str, k: int) -> list[dict]:
    """Top-k por relevancia de texto completo (BM25-like vía ts_rank en español)."""
    rows = conn.execute(
        """
        SELECT id, code, article_no, title, section, content,
               ts_rank(tsv, plainto_tsquery('spanish', %s)) AS score
        FROM articles
        WHERE tsv @@ plainto_tsquery('spanish', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, query, k),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_articles(conn: psycopg.Connection, code: str | None = None) -> int:
    if code:
        r = conn.execute("SELECT COUNT(*) FROM articles WHERE code = %s", (code,)).fetchone()
    else:
        r = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
    return int(r[0])


def _row_to_dict(r) -> dict:
    return {
        "id": r[0],
        "code": r[1],
        "article_no": r[2],
        "title": r[3],
        "section": r[4],
        "content": r[5],
        "score": float(r[6]) if r[6] is not None else 0.0,
    }
