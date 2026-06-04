"""add_document.py — Agrega un documento legal a LexIA Colombia (versión amigable).

Valida el PDF, muestra cuántos artículos detectó y un preview ANTES de subir nada,
lista los documentos ya cargados, pide confirmación y luego embebe e inserta
**lote a lote** (guardado incremental). Si se interrumpe (p. ej. por la cuota diaria),
NO pierdes el progreso: al volver a ejecutar, **reanuda** saltando lo ya guardado.

Uso:
    python add_document.py --pdf data/codigo_civil.pdf --code "C. Civil"
    python add_document.py --pdf data/codigo_civil.pdf --code "C. Civil" --yes        (sin preguntar)
    python add_document.py --pdf data/codigo_civil.pdf --code "C. Civil" --replace     (re-ingesta limpia)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")  # emojis/acentos en consola Windows
except Exception:  # noqa: BLE001
    pass

from rag.db import clear_code, connect, count_articles, ensure_schema, insert_articles
from rag.embeddings import embed_texts
from rag.ingest import extract_text, split_into_articles

CHUNK = 20          # artículos por lote (menos peticiones = menos gasto de cuota diaria)
THROTTLE = 6.0      # pausa entre lotes (segundos) para respetar el límite por minuto


def _existing_codes(conn) -> list[tuple[str, int]]:
    return conn.execute(
        "SELECT code, COUNT(*) FROM articles GROUP BY code ORDER BY code"
    ).fetchall()


def _existing_article_nos(conn, code: str) -> set[int]:
    rows = conn.execute("SELECT article_no FROM articles WHERE code = %s", (code,)).fetchall()
    return {r[0] for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description="Agrega un documento legal a LexIA Colombia.")
    p.add_argument("--pdf", required=True, help="Ruta al PDF del documento legal.")
    p.add_argument("--code", required=True, help='Etiqueta corta para citar, p.ej. "C. Civil".')
    p.add_argument("--yes", action="store_true", help="No pedir confirmación (modo automático).")
    p.add_argument("--replace", action="store_true",
                   help="Borra el código y lo re-ingesta desde cero (en vez de reanudar).")
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

    # 4) Resolver qué hay que subir (reanudar vs reemplazar)
    if args.replace:
        print(f"\n♻️  Modo --replace: se borrará '{args.code}' y se subirá completo de nuevo.")
        clear_code(conn, args.code)
        pending = articles
    else:
        done = _existing_article_nos(conn, args.code)
        pending = [a for a in articles if a["article_no"] not in done]
        if done:
            print(f"\n↩️  Reanudando '{args.code}': {len(done)} ya guardados, "
                  f"faltan {len(pending)}.")

    if not pending:
        print("\n✅ Nada que subir: este documento ya está completo en la base de datos.")
        conn.close()
        return

    # 5) Confirmación
    if not args.yes:
        resp = input(f"\n¿Subir {len(pending)} artículos como '{args.code}'? [s/N]: ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado. No se subió nada.")
            conn.close()
            sys.exit(0)

    # 6) Embeddings + inserción LOTE A LOTE (guardado incremental)
    print("\n⏳ Subiendo lote a lote (si se corta, el progreso se conserva)…")
    saved = 0
    try:
        for i in range(0, len(pending), CHUNK):
            chunk = pending[i : i + CHUNK]
            vectors = embed_texts([a["content"] for a in chunk], batch_size=CHUNK, throttle=0)
            for a, v in zip(chunk, vectors):
                a["embedding"] = v
            insert_articles(conn, chunk)          # ← se guarda este lote ya mismo
            saved += len(chunk)
            print(f"     guardados {saved}/{len(pending)}")
            if i + CHUNK < len(pending):
                time.sleep(THROTTLE)
    except Exception as e:  # noqa: BLE001
        total = count_articles(conn, args.code)
        conn.close()
        print(f"\n⚠️  Se interrumpió, pero se guardaron {saved} artículos en esta corrida "
              f"(total '{args.code}' en BD: {total}).")
        print("   Vuelve a ejecutar el MISMO comando para reanudar desde donde quedó.")
        print(f"   Detalle: {str(e)[:160]}")
        sys.exit(1)

    total_code = count_articles(conn, args.code)
    total_all = count_articles(conn)
    conn.close()
    print(f"\n🎉 Listo. '{args.code}': {total_code} artículos. Total en la BD: {total_all}.")
    print("   La demo en vivo (lexia-colombia.streamlit.app) ya puede responder sobre este documento.")


if __name__ == "__main__":
    main()
