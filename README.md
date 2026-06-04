# ⚖️ LexIA Colombia — Asistente Legal RAG

> Consulta la legislación colombiana en lenguaje natural y obtén respuestas **fundamentadas
> en el texto legal, con cita del artículo exacto**. Pensado para que abogados agilicen la
> investigación jurídica que hoy toma horas de lectura manual.

**🔗 Demo en vivo:** _(pega aquí tu URL desplegada)_
**🧪 Stack:** Python · pgvector (PostgreSQL/Supabase) · Google Gemini (`gemini-2.5-flash-lite`
+ `gemini-embedding-001`) · FastAPI · Streamlit
**💸 Costo:** $0 — usa el free tier de Google Gemini (generación + embeddings) y Supabase.
Los proveedores son intercambiables (Anthropic / OpenAI) vía variables de entorno.
**🛡️ Robustez:** manejo de límites del free tier (throttling + retry con backoff ante 429/503).

![demo](docs/demo.gif) <!-- opcional: graba un GIF de 10s del chat respondiendo -->

---

## ¿Qué problema resuelve?

Un abogado que necesita saber *cómo se liquida una indemnización por despido sin justa
causa* hoy abre el código, busca el artículo, lo lee y lo interpreta. **LexIA** hace ese
recorrido en segundos y, lo más importante, **cita la fuente** para que el profesional la
verifique. No reemplaza el criterio jurídico: lo acelera.

## ¿Por qué es técnicamente sólido (y no un "wrapper de ChatGPT")?

| Decisión | Qué hace | Por qué importa |
|---|---|---|
| **Chunking por artículo** | Parte la ley por su unidad natural (Art. N), no por bloques de 500 tokens | Cada cita corresponde a un artículo real y completo |
| **Búsqueda híbrida + RRF** | Combina embeddings (significado) con texto completo (término exacto) | En derecho, *"justa causa"* o *"fuero"* deben matchear literalmente |
| **Generación grounded** | El modelo solo puede usar los artículos recuperados | Reduce alucinación al mínimo |
| **Guardrail explícito** | Si no hay respaldo, responde "no encuentro fuente" en vez de inventar | Crítico en un dominio donde alucinar es inaceptable |
| **Evaluación con métricas** | recall@k del retrieval + faithfulness (LLM-as-judge) | Permite mejorar con datos, no a ojo |
| **Disclaimer legal** | Cada respuesta aclara que no es asesoría jurídica | Responsabilidad profesional |
| **Failover de modelos** | Si un modelo agota su cuota del free tier, salta al siguiente automáticamente | Resiliencia en producción con presupuesto $0 |

## Arquitectura

```
        PDF legal (dominio público)
                │
   ┌────────────▼─────────────┐
   │  INGESTA (rag/ingest.py)  │  parse → chunk por artículo → embeddings → pgvector
   └────────────┬─────────────┘
                │
 Pregunta ─► ┌──▼───────────────────────────────────┐
             │  RETRIEVAL (rag/retriever.py)         │
             │  semántica (pgvector <=>) + léxica    │
             │  (ts_rank) → fusión RRF → top-k       │
             └──┬───────────────────────────────────┘
                │ contexto + citas
             ┌──▼───────────────────────────────────┐
             │  GENERACIÓN (rag/generator.py)        │
             │  Claude + prompt grounded + guardrail │
             └──┬───────────────────────────────────┘
                │
   Respuesta con (Art. N) + disclaimer
       │                          │
  FastAPI (api/main.py)     Streamlit (app.py)
```

## Estructura del proyecto

```
lexia-colombia/
├── rag/
│   ├── config.py        # configuración desde .env
│   ├── db.py            # pgvector: esquema, inserción, búsquedas
│   ├── embeddings.py    # OpenAI o bge-m3 local (intercambiable)
│   ├── ingest.py        # PDF → artículos → embeddings → DB
│   ├── retriever.py     # búsqueda híbrida + Reciprocal Rank Fusion
│   ├── generator.py     # generación grounded + guardrail + streaming
│   └── prompts.py       # system prompt anti-alucinación
├── api/main.py          # FastAPI (/chat, /chat/stream, /health)
├── app.py               # UI de chat en Streamlit
├── eval/                # dataset + métricas (recall@k, faithfulness)
├── scripts/setup_db.sql # esquema de referencia
└── data/                # PDFs fuente (no versionados)
```

## Puesta en marcha

### 1. Requisitos
- Python 3.11+
- Una base PostgreSQL con `pgvector` (recomendado: **Supabase**, free tier)
- Una **API key gratis de Google AI Studio** (sin tarjeta): https://aistudio.google.com/apikey

### 2. Instalación
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # y rellena tus llaves
```

### 3. Indexar un código legal
Coloca el PDF en `data/` (ver [data/README.md](data/README.md)) y ejecuta:
```bash
python -m rag.ingest --pdf data/codigo_sustantivo_trabajo.pdf --code CST
```

### 4. Probar
```bash
streamlit run app.py            # UI de chat
# o la API:
uvicorn api.main:app --reload   # http://localhost:8000/docs
```

### 5. Evaluar
```bash
python -m eval.evaluate --k 8
```
Resultados medidos sobre el Código Sustantivo del Trabajo (512 artículos, 10 preguntas):

| Métrica | Valor | Qué significa |
|---|---|---|
| **Recall@8** | **90 %** (9/10) | En el 90 % de los casos, el artículo correcto está entre los 8 recuperados |
| **Faithfulness** | **90 %** (9/10) | El 90 % de las respuestas se apoyan solo en el texto recuperado (juez LLM) |

> El recall sube de 70 % (k=6) a 90 % (k=8) y se estabiliza; por eso `TOP_K=8`.

## Despliegue (demo pública gratis)
- **Base de datos:** Supabase (PostgreSQL + pgvector administrado).
- **App:** Streamlit Community Cloud (todo Python) o Render / Hugging Face Spaces.
- Configura las variables de entorno (`DATABASE_URL`, `ANTHROPIC_API_KEY`, …) en el panel
  del proveedor. **Nunca** subas el `.env`.

## Roadmap
- [ ] Reranking con cross-encoder (`bge-reranker`) para subir la precisión del top-k.
- [ ] Más códigos (Civil, Constitución, Penal).
- [ ] Resaltado del fragmento exacto citado dentro del artículo.
- [ ] Historial conversacional con reescritura de la consulta (multi-turn).

---

⚠️ **Aviso:** LexIA es un asistente informativo y **no constituye asesoría jurídica**.
Verifica siempre las fuentes citadas.

_Autor: José Sierra · [LinkedIn](https://www.linkedin.com/in/jose-sierra98/) ·
[GitHub](https://github.com/jost9804)_
