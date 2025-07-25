"""Retriever agent that fetches context for a query."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..retrieval.vector_store import get_retriever
from ..config import settings


@dataclass
class RetrieverAgent:
    """Callable node compatible with LangGraph that enriches state with context."""

    retriever = get_retriever(top_k=settings.TOP_K)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query: str = state["query"]
        docs = self.retriever.get_relevant_documents(query)
        state["context"] = [d.page_content for d in docs]
        return state 