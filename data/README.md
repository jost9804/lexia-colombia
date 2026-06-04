# Datos fuente

Coloca aquí el PDF del código legal a indexar. Las leyes colombianas son de **dominio
público**.

## Código inicial sugerido: Código Sustantivo del Trabajo (CST)

1. Descarga el PDF oficial desde una fuente pública, por ejemplo:
   - Secretaría del Senado — gestor normativo (`secretariasenado.gov.co`)
   - Función Pública — gestor normativo (`funcionpublica.gov.co`)
2. Guárdalo aquí como `codigo_sustantivo_trabajo.pdf`.
3. Ingéstalo. **Forma recomendada** (valida y muestra un preview antes de subir):

   ```bash
   python add_document.py --pdf data/codigo_sustantivo_trabajo.pdf --code "CST"
   ```

   O la forma directa, sin confirmación:

   ```bash
   python -m rag.ingest --pdf data/codigo_sustantivo_trabajo.pdf --code CST
   ```

> Los `*.pdf` están en `.gitignore` para no versionar archivos pesados. El repo guarda el
> código, no los documentos.

## Añadir más códigos después

El mismo pipeline sirve para cualquier código con artículos numerados:

```bash
python -m rag.ingest --pdf data/codigo_civil.pdf --code "C. Civil"
python -m rag.ingest --pdf data/constitucion.pdf --code "Const."
```

Si el patrón de "Artículo N" del documento difiere, ajusta `ARTICLE_RE` en
[`rag/ingest.py`](../rag/ingest.py).
