"""UI de chat en Streamlit. Llama al core RAG directamente (deploy todo-en-uno).

Ejecutar:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

# Puente: en Streamlit Cloud las llaves vienen de st.secrets; en local, de .env.
# Copiamos los secrets a variables de entorno ANTES de importar el core RAG.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 - sin secrets locales: se usa .env
    pass

from rag.generator import answer_stream, retrieve

st.set_page_config(page_title="LexIA Colombia", page_icon="⚖️", layout="centered")

st.title("⚖️ LexIA Colombia")
st.caption(
    "Asistente de investigación jurídica sobre legislación colombiana. "
    "Respuestas fundamentadas en el texto legal, con cita del artículo."
)

with st.sidebar:
    st.header("Acerca de")
    st.markdown(
        "- **RAG** sobre artículos de la ley colombiana.\n"
        "- Búsqueda **híbrida** (semántica + léxica).\n"
        "- **Citas verificables** y guardrail anti-alucinación.\n"
    )
    top_k = st.slider("Artículos a recuperar (k)", 3, 12, 6)
    st.warning("Asistente informativo. No constituye asesoría jurídica.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Historial.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Escribe tu consulta jurídica…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Recuperamos una sola vez y reutilizamos para la generación (ahorra embeddings).
        sources = retrieve(prompt, k=top_k)
        if sources:
            with st.expander(f"📚 Fuentes recuperadas ({len(sources)})"):
                for s in sources:
                    title = f" — {s['title']}" if s.get("title") else ""
                    st.markdown(f"**Art. {s['article_no']} {s['code']}**{title}")
                    st.caption(s["content"][:300] + ("…" if len(s["content"]) > 300 else ""))

        # Respuesta en streaming, reutilizando las fuentes ya recuperadas.
        full = st.write_stream(answer_stream(prompt, k=top_k, chunks=sources))

    st.session_state.messages.append({"role": "assistant", "content": full})
