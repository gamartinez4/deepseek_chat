# Product Query RAG Bot

A RAG (Retrieval Augmented Generation) system that answers queries about products using a vector store and LLM.

## Features

- Product document indexing with FAISS vector store
- Query answering using LangChain and LangGraph
- FastAPI server with GitHub token authentication
- Containerized deployment support

## Requirements

- Python 3.9+
- Docker (optional)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
Create a `.env` file with custom settings:
```env
ENDPOINT=https://models.github.ai/inference
MODEL_NAME=deepseek/DeepSeek-V3-0324
TOP_K=3
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
INDEX_PATH=store/faiss_index
REQUEST_TIMEOUT=30
```

## Indexing Documents

Before querying, you need to index your product documents:

```bash
python -m src.index_documents --docs-path data/ --index-path store/faiss_index
```

This will:
1. Read all .txt files from the docs-path directory
2. Create embeddings using the specified model
3. Store the FAISS index in the index-path directory

## Running the Server

### Local Development

```bash
uvicorn src.server:app --reload
```

### Production

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
docker build -t product-rag-bot .
docker run -p 8000:8000 product-rag-bot
```

## API Usage

The server exposes a `/query` endpoint that accepts POST requests:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "query": "Tell me about product A"}'
```

## Testing

Run the test suite:

```bash
pytest
```

## Project Structure

- `data/`: Product description documents
- `src/`: Source code
  - `agents/`: RAG system agents
  - `retrieval/`: Vector store implementation
  - `server.py`: FastAPI application
  - `index_documents.py`: Document indexing utility
- `store/`: FAISS index storage
- `tests/`: Test suite 