# Product-Query Bot via RAG Pipeline

Microservicio en Python que expone un endpoint REST para responder preguntas sobre productos mediante un pipeline RAG (Retrieval-Augmented Generation) implementado con un enfoque multi-agente usando LangGraph.

## Características

* **/query** – endpoint `POST` que recibe `user_id` y `query`.
* **Pipeline RAG** – recuperación semántica con FAISS + HuggingFace embeddings y generación con el modelo `deepseek/DeepSeek-V3-0324`.
* **Multi-agente** – `RetrieverAgent` y `ResponderAgent` coordinados mediante LangGraph.
* **Configuración** – variables de entorno (`API_TOKEN`, `TOP_K`, etc.).
* **Tests** – suite con `pytest`.
* **Docker** – lista para contenedorización.

## Instalación rápida

```bash
git clone <repo_url>
cd ejemplo
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Indexar documentos

Coloca archivos `.txt` con descripciones de producto dentro de `data/` y ejecuta:

```bash
python -m src.index_documents --docs-path data/
```

### Ejecutar servicio

```bash
export API_TOKEN="TU_TOKEN"
uvicorn src.server:app --reload
```

Prueba con:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"user_id":"123","query":"¿Este producto es azul?"}'
```

### Docker

```bash
docker build -t product-query-bot .
docker run -e API_TOKEN="TU_TOKEN" -p 8000:8000 product-query-bot
```

## Variables de entorno

| Variable            | Default | Descripción                                   |
|---------------------|---------|-----------------------------------------------|
| `API_TOKEN`         | (vacío) | Token para acceder al endpoint LLM            |
| `TOP_K`             | `3`     | Nº de documentos a recuperar                  |
| `INDEX_PATH`        | `store/faiss_index` | Carpeta donde se guarda el índice |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Modelo de embeddings |

## Tests

```bash
pytest -q
```

## Tiempo empleado

<especificar tiempo invertido>
