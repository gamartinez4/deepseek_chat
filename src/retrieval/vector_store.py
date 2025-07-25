"""Vector store utilities using FAISS and HuggingFace embeddings via LangChain."""
from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:  # fallback para evitar fallo si el paquete no está instalado
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS

from ..config import settings

_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)


def build_index(texts: List[str], index_path: str | Path | None = settings.INDEX_PATH) -> None:
    """Build a FAISS index from raw texts and persist it to disk.

    Si ``index_path`` es ``None`` se usará la ruta por defecto definida en settings.
    """
    if index_path is None:
        index_path = settings.INDEX_PATH

    db = FAISS.from_texts(texts=texts, embedding=_embeddings)
    db.save_local(str(index_path))


def get_retriever(index_path: str | Path = settings.INDEX_PATH, top_k: int = settings.TOP_K):
    """Load a FAISS index and return an initialized retriever.
    Si no existe el índice o falla la deserialización segura, se lanza un error
    con un mensaje claro para que el usuario ejecute el script de indexado.
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No se encontró el índice en '{index_path}'. Ejecuta 'python -m src.index_documents' primero."
        )

    try:
        db = FAISS.load_local(
            str(index_path), embeddings=_embeddings, allow_dangerous_deserialization=True
        )
    except ValueError as exc:
        # Posible incompatibilidad o corrupción; proporcionar una sugerencia útil.
        raise RuntimeError(
            "Error cargando el índice FAISS. Si confías en el archivo, "
            "puedes regenerarlo ejecutando el script de indexado."
        ) from exc
    return db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def retrieve(query: str, index_path: str | Path = settings.INDEX_PATH, top_k: int = settings.TOP_K) -> List[str]:
    """Convenience helper to directly obtain relevant documents' raw text."""
    retriever = get_retriever(index_path=index_path, top_k=top_k)
    docs = retriever.get_relevant_documents(query)
    return [d.page_content for d in docs] 