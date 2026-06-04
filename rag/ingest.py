"""Pipeline de ingesta: PDF legal → artículos → embeddings → pgvector.

Chunking *consciente del dominio*: en vez de cortar cada 500 tokens, partimos por
ARTÍCULO, que es la unidad natural de citación jurídica. Cada artículo conserva su
número y la sección (libro/título/capítulo) en la que está, para citar con precisión.

Uso:
    python -m rag.ingest --pdf data/codigo_sustantivo_trabajo.pdf --code CST
"""
from __future__ import annotations

import argparse
import re

import pdfplumber

from .db import clear_code, connect, count_articles, ensure_schema, insert_articles
from .embeddings import embed_texts

# "ARTÍCULO 64.", "Articulo 64", "ART. 64" — toleramos acentos y mayúsculas.
ARTICLE_RE = re.compile(r"\bART[IÍ]?CULO?\.?\s+(\d+)\b", re.IGNORECASE)
# Encabezados de sección que queremos rastrear como metadata.
SECTION_RE = re.compile(r"^\s*(LIBRO|T[IÍ]TULO|CAP[IÍ]TULO)\b.*$", re.IGNORECASE | re.MULTILINE)


def extract_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def split_into_articles(text: str, code: str) -> list[dict]:
    """Divide el texto en artículos con su sección y un título tentativo."""
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        raise ValueError(
            "No se detectaron artículos con el patrón esperado. "
            "Revisa el PDF o ajusta ARTICLE_RE."
        )

    articles: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Sección vigente = último encabezado de sección antes de este artículo.
        section = _last_section_before(text, start)
        # Título tentativo = primera línea tras "Artículo N." (a veces es el epígrafe).
        title = _guess_title(body)

        content = _clean(body)
        if len(content) < 20:  # descarta falsos positivos (referencias cruzadas)
            continue

        articles.append(
            {
                "code": code,
                "article_no": int(m.group(1)),
                "title": title,
                "section": section,
                "content": content,
            }
        )
    return _dedupe_by_article(articles)


def _last_section_before(text: str, pos: int) -> str | None:
    section = None
    for sm in SECTION_RE.finditer(text):
        if sm.start() > pos:
            break
        section = sm.group(0).strip()
    return section


def _guess_title(body: str) -> str | None:
    # Quita el encabezado "Artículo N." y toma una frase corta como epígrafe.
    rest = ARTICLE_RE.sub("", body, count=1).strip(" .:-\n")
    first_line = rest.split("\n", 1)[0].strip()
    if 3 <= len(first_line) <= 90 and first_line[:1].isupper():
        return first_line.rstrip(". ")
    return None


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_by_article(articles: list[dict]) -> list[dict]:
    """Si un número de artículo aparece repetido, conserva el fragmento más largo."""
    best: dict[int, dict] = {}
    for a in articles:
        n = a["article_no"]
        if n not in best or len(a["content"]) > len(best[n]["content"]):
            best[n] = a
    return [best[k] for k in sorted(best)]


def ingest(pdf_path: str, code: str) -> int:
    print(f"[1/4] Extrayendo texto de {pdf_path} …")
    text = extract_text(pdf_path)

    print("[2/4] Partiendo en artículos …")
    articles = split_into_articles(text, code)
    print(f"      {len(articles)} artículos detectados.")

    print("[3/4] Generando embeddings …")
    # Lotes grandes (menos peticiones → menos gasto de la cuota diaria) con pausa
    # entre lotes para respetar el límite por minuto. Para grandes volúmenes con
    # reanudación automática, usa el script add_document.py.
    vectors = embed_texts(
        [a["content"] for a in articles],
        batch_size=20,
        throttle=6.0,
        verbose=True,
    )
    for a, v in zip(articles, vectors):
        a["embedding"] = v

    print("[4/4] Guardando en pgvector …")
    conn = connect()
    ensure_schema(conn)
    clear_code(conn, code)
    insert_articles(conn, articles)
    total = count_articles(conn, code)
    conn.close()
    print(f"✓ Listo. {total} artículos del código {code} indexados.")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Ingesta de un código legal en pgvector.")
    p.add_argument("--pdf", required=True, help="Ruta al PDF del código legal.")
    p.add_argument("--code", required=True, help='Identificador corto, p.ej. "CST".')
    args = p.parse_args()
    ingest(args.pdf, args.code)


if __name__ == "__main__":
    main()
