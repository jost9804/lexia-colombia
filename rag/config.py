"""Configuración central leída desde variables de entorno (.env).

Proveedores intercambiables:
  - Generación  (GEN_PROVIDER):   "gemini" (gratis) | "anthropic"
  - Embeddings  (EMBED_PROVIDER): "gemini" (gratis) | "openai" | "local"
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno {name}. "
            f"Copia .env.example a .env y rellénala."
        )
    return value


# Dimensión del vector por modelo de embedding (debe coincidir con la tabla).
EMBED_DIMS = {
    "text-embedding-004": 768,      # Gemini (gratis)
    "text-embedding-3-small": 1536,  # OpenAI
    "BAAI/bge-m3": 1024,             # local
}


@dataclass(frozen=True)
class Config:
    database_url: str
    # Generación
    gen_provider: str
    gen_model: str
    gen_models: list[str]
    google_api_key: str | None
    anthropic_api_key: str | None
    # Embeddings
    embed_provider: str
    gemini_embed_model: str
    openai_api_key: str | None
    openai_embed_model: str
    local_embed_model: str
    embed_dim: int
    # Recuperación
    top_k: int

    @classmethod
    def load(cls) -> "Config":
        gen_provider = os.getenv("GEN_PROVIDER", "gemini").lower()
        embed_provider = os.getenv("EMBED_PROVIDER", "gemini").lower()
        google_key = os.getenv("GOOGLE_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Validaciones de coherencia.
        if gen_provider == "gemini" and not google_key:
            raise RuntimeError("GEN_PROVIDER=gemini requiere GOOGLE_API_KEY.")
        if gen_provider == "anthropic" and not anthropic_key:
            raise RuntimeError("GEN_PROVIDER=anthropic requiere ANTHROPIC_API_KEY.")
        if embed_provider == "gemini" and not google_key:
            raise RuntimeError("EMBED_PROVIDER=gemini requiere GOOGLE_API_KEY.")
        if embed_provider == "openai" and not openai_key:
            raise RuntimeError("EMBED_PROVIDER=openai requiere OPENAI_API_KEY.")

        gemini_embed_model = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        local_embed_model = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

        # Resuelve la dimensión automáticamente según el modelo activo.
        active_embed_model = {
            "gemini": gemini_embed_model,
            "openai": openai_embed_model,
            "local": local_embed_model,
        }[embed_provider]
        embed_dim = int(os.getenv("EMBED_DIM", "0")) or EMBED_DIMS.get(active_embed_model, 768)

        # Cadena de modelos para failover: GEN_MODELS (coma-separado) o GEN_MODEL único.
        gen_model = os.getenv("GEN_MODEL", "gemma-4-31b-it")
        gen_models = [m.strip() for m in os.getenv("GEN_MODELS", "").split(",") if m.strip()]
        if not gen_models:
            gen_models = [gen_model]

        return cls(
            database_url=_require("DATABASE_URL"),
            gen_provider=gen_provider,
            gen_model=gen_models[0],
            gen_models=gen_models,
            google_api_key=google_key,
            anthropic_api_key=anthropic_key,
            embed_provider=embed_provider,
            gemini_embed_model=gemini_embed_model,
            openai_api_key=openai_key,
            openai_embed_model=openai_embed_model,
            local_embed_model=local_embed_model,
            embed_dim=embed_dim,
            top_k=int(os.getenv("TOP_K", "6")),
        )


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config.load()
    return _cfg
