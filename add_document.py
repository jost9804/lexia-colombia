"""add_document.py — Agrega un documento legal a LexIA Colombia (versión amigable).

Valida el PDF, muestra cuántos artículos detectó y un preview ANTES de subir nada,
muestra qué documentos ya están cargados, pide confirmación y luego genera los
embeddings e inserta en pgvector.

Uso:
    python add_document.py --pdf data/codigo_civil.pdf --code "C. Civil"
    python add_document.py --pdf data/codigo_civil.pdf --code "C. Civil" --yes   (sin preguntar)
"""
from __future__ import annotations

import argparse
import os
import sys

# Evita errores de codificación de emojis/acentos en la consola de Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from rag.db import clear_code, connect, count_articles, ensure_schema, insert_articles
from rag.embeddings import embed_texts
from rag.ingest import extract_text, split_into_articles


def _existing_codes(conn) -> list[tuple[str, int]]:
    return conn.execute(
        "SELECT code, COUNT(*) FROM articles GROUP BY code ORDER BY code"
    ).fetchall()


def main() -> None:
    p = argparse.ArgumentParser(description="Agrega un documento legal a LexIA Colombia.")
    p.add_argument("--pdf", required=True, help="Ruta al PDF del documento legal.")
    p.add_argument("--code", required=True, help='Etiqueta corta para citar, p.ej. "C. Civil".')
    p.add_argument("--yes", action="store_true", help="No pedir confirmación (modo automático).")
    args = p.parse_args()

    # 1) Validar el archivo
    if not os.path.isfile(args.pdf):
        print(f"❌ No existe el archivo: {args.pdf}")
        sys.exit(1)

    # 2) Parsear y detectar artículos (sin tocar la base de datos todavía)
    print(f"📄 Leyendo {args.pdf} …")
    text = extract_text(args.pdf)
    try:
        articles = split_into_articles(text, args.code)
    except ValueError:
        print("❌ No se detectaron artículos con el patrón 'ARTÍCULO N'.")
        print("   Posibles causas: el PDF está escaneado (sin texto) o usa otro formato de")
        print("   numeración. Solución: ajusta ARTICLE_RE en rag/ingest.py y vuelve a intentar.")
        sys.exit(1)

    nums = [a["article_no"] for a in articles]
    print(f"\n✅ Detectados {len(articles)} artículos (del Art. {min(nums)} al Art. {max(nums)}).")
    print("   Vista previa:")
    for a in articles[:3]:
        title = f" — {a['title']}" if a.get("title") else ""
        snippet = " ".join(a["content"][:90].split())
        print(f"     • Art. {a['article_no']} {args.code}{title}: {snippet}…")

    # 3) Estado actual de la base de datos
    conn = connect()
    ensure_schema(conn)
    existing = _existing_codes(conn)
    if existing:
        print("\n📚 Documentos ya cargados en la base de datos:")
        for code, n in existing:
            print(f"     • {code}: {n} artículos")
    if any(code == args.code for code, _ in existing):
        print(f"\n⚠️  El código '{args.code}' YA existe y será REEMPLAZADO por este documento.")

    # 4) Confirmación
    if not args.yes:
        resp = input(f"\n¿Subir {len(articles)} artículos como '{args.code}'? [s/N]: ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado. No se subió nada.")
            conn.close()
            sys.exit(0)

    # 5) Embeddings (ritmo conservador para el free tier) + inserción
    print("\n⏳ Generando embeddings (free tier: lotes pequeños con pausas; puede tardar varios minutos)…")
    vectors = embed_texts(
        [a["content"] for a in articles], batch_size=5, throttle=12.0, verbose=True
    )
    for a, v in zip(articles, vectors):
        a["embedding"] = v

    print("💾 Guardando en la base de datos…")
    clear_code(conn, args.code)
    insert_articles(conn, articles)
    total_code = count_articles(conn, args.code)
    total_all = count_articles(conn)
    conn.close()

    print(f"\n🎉 Listo. '{args.code}': {total_code} artículos indexados. Total en la BD: {total_all}.")
    print("   La demo en vivo (lexia-colombia.streamlit.app) ya puede responder sobre este documento.")


if __name__ == "__main__":
    main()
