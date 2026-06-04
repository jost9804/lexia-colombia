"""Generación grounded: recupera artículos, construye contexto y responde con citas.

Proveedor de generación (GEN_PROVIDER): "gemini" (gratis) | "anthropic".

Failover de modelos: GEN_MODELS define una cadena de modelos. Cuando uno agota su cuota
diaria del free tier (429), el sistema pasa automáticamente al siguiente. Así se multiplica
la disponibilidad gratuita sin tarjeta.

Guardrails:
  - Si el retriever no devuelve nada, ni siquiera llamamos al modelo (cero costo, cero
    riesgo de alucinación).
  - El system prompt obliga a citar y a declarar cuando no hay respaldo.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Iterator

from .config import get_config
from .prompts import SYSTEM_PROMPT, build_context_block
from .retriever import hybrid_search

_RETRYABLE = (
    "429", "RESOURCE_EXHAUSTED",      # cuota agotada
    "503", "UNAVAILABLE", "overloaded",  # modelo sobrecargado
    "500", "INTERNAL",               # error transitorio del servidor
)

NO_CONTEXT_MSG = (
    "No encuentro respaldo suficiente en las fuentes cargadas para responder esto "
    "con certeza.\n\n⚠️ Asistente informativo. No constituye asesoría jurídica."
)

RATE_LIMIT_MSG = (
    "⏳ **Límite del free tier alcanzado en todos los modelos disponibles.** Los "
    "**artículos recuperados arriba sí son correctos**; solo falta la redacción del "
    "modelo. Intenta de nuevo en unos minutos o mañana (la cuota gratuita se resetea a "
    "diario).\n\n⚠️ Asistente informativo. No constituye asesoría jurídica."
)


def _is_retryable(err: Exception) -> bool:
    return any(t in str(err) for t in _RETRYABLE)


def _with_retry(fn, *args, max_retries: int = 3):
    """Reintenta una llamada ante 429/503 transitorios (límite por minuto, sobrecarga)."""
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            time.sleep(15 * (attempt + 1))


def _user_message(query: str, chunks: list[dict]) -> str:
    return (
        f"CONTEXTO (artículos recuperados):\n\n{build_context_block(chunks)}\n\n"
        f"PREGUNTA DEL USUARIO:\n{query}"
    )


def retrieve(query: str, k: int | None = None) -> list[dict]:
    return hybrid_search(query, k=k)


# ─────────────────────────── Gemini / Gemma (gratis) ───────────────────────────
@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=get_config().google_api_key)


def _is_gemma(model: str) -> bool:
    return "gemma" in model.lower()


def _gemini_gcfg(system: str, model: str):
    """Gemma no soporta system_instruction ni thinking → config mínima.
    En modelos 2.5 desactivamos 'thinking' (evita respuestas vacías y ahorra cuota)."""
    from google.genai import types

    if _is_gemma(model):
        return types.GenerateContentConfig(max_output_tokens=1024)
    kwargs = dict(system_instruction=system, max_output_tokens=1024)
    if "2.5" in model:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _gemini_contents(user_msg: str, system: str, model: str) -> str:
    # Gemma no acepta system aparte: lo anteponemos al contenido.
    return f"{system}\n\n{user_msg}" if _is_gemma(model) else user_msg


def _gemini_stream(user_msg: str, system: str, model: str) -> Iterator[str]:
    stream = _gemini_client().models.generate_content_stream(
        model=model,
        contents=_gemini_contents(user_msg, system, model),
        config=_gemini_gcfg(system, model),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def _gemini_complete(user_msg: str, system: str, model: str) -> str:
    resp = _gemini_client().models.generate_content(
        model=model,
        contents=_gemini_contents(user_msg, system, model),
        config=_gemini_gcfg(system, model),
    )
    return resp.text or ""


# ─────────────────────────── Anthropic (de pago) ───────────────────────────
@lru_cache(maxsize=1)
def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic(api_key=get_config().anthropic_api_key)


def _anthropic_stream(user_msg: str, system: str, model: str) -> Iterator[str]:
    with _anthropic_client().messages.stream(
        model=model, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        yield from stream.text_stream


def _anthropic_complete(user_msg: str, system: str, model: str) -> str:
    msg = _anthropic_client().messages.create(
        model=model, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


_STREAM = {"gemini": _gemini_stream, "anthropic": _anthropic_stream}
_COMPLETE = {"gemini": _gemini_complete, "anthropic": _anthropic_complete}


# ─────────────────────────── API pública ───────────────────────────
def answer_stream(
    query: str, k: int | None = None, chunks: list[dict] | None = None
) -> Iterator[str]:
    """Respuesta token a token con failover de modelos. Si todos agotan cuota, emite un
    mensaje claro (no un error). Si se pasan `chunks`, evita recuperar de nuevo."""
    if chunks is None:
        chunks = retrieve(query, k=k)
    if not chunks:
        yield NO_CONTEXT_MSG
        return

    cfg = get_config()
    stream_fn = _STREAM[cfg.gen_provider]
    user_msg = _user_message(query, chunks)

    for model in cfg.gen_models:
        started = False
        try:
            for token in stream_fn(user_msg, SYSTEM_PROMPT, model):
                started = True
                yield token
            return  # éxito
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e):
                raise
            if started:
                # Ya emitimos texto parcial; no podemos cambiar de modelo limpiamente.
                yield "\n\n" + RATE_LIMIT_MSG
                return
            continue  # este modelo está sin cuota → probar el siguiente
    yield RATE_LIMIT_MSG  # todos los modelos agotados


def _complete_with_failover(user_msg: str, system: str) -> str:
    """Completa probando cada modelo de la cadena hasta que uno responda."""
    cfg = get_config()
    complete_fn = _COMPLETE[cfg.gen_provider]
    last_err: Exception | None = None
    for model in cfg.gen_models:
        try:
            return _with_retry(complete_fn, user_msg, system, model)
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e):
                raise
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Sin modelos disponibles")


def llm_complete(prompt: str, system: str = "") -> str:
    """Completa un prompt (sin recuperación). Útil para el juez de evaluación."""
    return _complete_with_failover(prompt, system)


def answer(query: str, k: int | None = None) -> dict:
    """Versión no-streaming: devuelve texto + las citas usadas (para eval/API)."""
    chunks = retrieve(query, k=k)
    if not chunks:
        return {"answer": NO_CONTEXT_MSG, "citations": []}
    try:
        text = _complete_with_failover(_user_message(query, chunks), SYSTEM_PROMPT)
    except Exception as e:  # noqa: BLE001
        if _is_retryable(e):
            text = RATE_LIMIT_MSG
        else:
            raise
    citations = [
        {"article_no": c["article_no"], "code": c["code"], "title": c.get("title")}
        for c in chunks
    ]
    return {"answer": text, "citations": citations}
