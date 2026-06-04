"""Capa de embeddings con proveedores intercambiables:
  - "gemini" → text-embedding-004 (GRATIS, recomendado, 768 dim)
  - "openai" → text-embedding-3-small (de pago)
  - "local"  → bge-m3 vía sentence-transformers (gratis pero pesado)
"""
from __future__ import annotations

import time
from functools import lru_cache

from .config import get_config


def _is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


# ─────────────────────────── Gemini (gratis) ───────────────────────────
@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=get_config().google_api_key)


def _embed_gemini(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    from google.genai import types

    cfg = get_config()
    resp = _gemini_client().models.embed_content(
        model=cfg.gemini_embed_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=cfg.embed_dim
        ),
    )
    return [e.values for e in resp.embeddings]


# ─────────────────────────── OpenAI ───────────────────────────
@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=get_config().openai_api_key)


def _embed_openai(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    cfg = get_config()  # task_type no aplica a OpenAI; se ignora.
    resp = _openai_client().embeddings.create(model=cfg.openai_embed_model, input=texts)
    return [d.embedding for d in resp.data]


# ─────────────────────────── Local (bge-m3) ───────────────────────────
@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_config().local_embed_model)


def _embed_local(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    vecs = _local_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


_PROVIDERS = {
    "gemini": _embed_gemini,
    "openai": _embed_openai,
    "local": _embed_local,
}


# ─────────────────────────── API pública ───────────────────────────
def embed_texts(
    texts: list[str],
    batch_size: int = 20,
    task_type: str = "RETRIEVAL_DOCUMENT",
    throttle: float = 0.0,
    max_retries: int = 8,
    verbose: bool = False,
) -> list[list[float]]:
    """Embebe una lista de textos por lotes, con throttling y reintentos ante 429.

    Los free tiers limitan peticiones/tokens por minuto; espaciamos los lotes y, si llega
    un 429, esperamos (backoff) y reintentamos. Así la ingesta completa sin pagar nada.
    """
    embed = _PROVIDERS[get_config().embed_provider]
    out: list[list[float]] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for bi, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                out.extend(embed(batch, task_type))
                break
            except Exception as e:  # noqa: BLE001
                if not _is_rate_limit(e) or attempt == max_retries - 1:
                    raise
                # Límite DIARIO: reintentar hoy no sirve → fallar rápido.
                if "PerDay" in str(e):
                    print("      [cuota diaria agotada] Se acabaron las ~1000 peticiones de "
                          "embeddings de hoy. Vuelve a ejecutar tras el reset (medianoche "
                          "hora del Pacifico, ~2 a.m. en Colombia). El progreso ya guardado se conserva.")
                    raise
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s...
                print(f"      [throttle] Limite por minuto; esperando {wait}s y reintentando...")
                time.sleep(wait)
        if verbose:
            print(f"      embebidos {min(i + batch_size, len(texts))}/{len(texts)}")
        if bi < n_batches - 1 and throttle:
            time.sleep(throttle)  # espaciar lotes para no saturar el limite por minuto
    return out


def embed_query(text: str) -> list[float]:
    # task_type de consulta mejora el emparejamiento consulta↔documento en Gemini.
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
