-- Esquema de referencia para LexIA Colombia (pgvector).
-- El pipeline de ingesta crea esto automáticamente con la dimensión correcta,
-- pero puedes ejecutarlo manualmente en el SQL editor de Supabase si lo prefieres.
-- NOTA: la dimensión depende del modelo de embedding:
--   Gemini text-embedding-004 = 768  ·  OpenAI 3-small = 1536  ·  bge-m3 = 1024
-- Ajusta vector(768) abajo si cambias de modelo.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id          BIGSERIAL PRIMARY KEY,
    code        TEXT    NOT NULL,            -- p.ej. "CST"
    article_no  INTEGER,                     -- número de artículo
    title       TEXT,                        -- epígrafe del artículo
    section     TEXT,                        -- libro / título / capítulo
    content     TEXT    NOT NULL,            -- texto completo del artículo
    embedding   vector(768),                 -- 768 = Gemini text-embedding-004
    tsv         tsvector
);

-- Índice vectorial (HNSW, no requiere entrenamiento) para búsqueda semántica por coseno.
CREATE INDEX IF NOT EXISTS articles_embedding_idx
    ON articles USING hnsw (embedding vector_cosine_ops);

-- Índice de texto completo en español para búsqueda léxica.
CREATE INDEX IF NOT EXISTS articles_tsv_idx
    ON articles USING gin (tsv);
