"""API FastAPI: expone el RAG como servicio con streaming SSE.

Ejecutar:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.db import connect, count_articles
from rag.generator import answer, answer_stream

app = FastAPI(title="LexIA Colombia", version="1.0.0")

# CORS abierto para el demo; restríngelo en producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: int | None = None


@app.get("/health")
def health() -> dict:
    try:
        conn = connect()
        n = count_articles(conn)
        conn.close()
        return {"status": "ok", "articles_indexed": n}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": str(e)}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Respuesta completa con citas (JSON)."""
    return answer(req.question, k=req.k)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Respuesta en streaming (text/event-stream)."""

    def event_gen():
        for token in answer_stream(req.question, k=req.k):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
