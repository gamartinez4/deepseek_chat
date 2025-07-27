"""LangGraph pipeline wiring Retriever and Responder agents."""
from __future__ import annotations

# typing_extensions offers TypedDict, NotRequired for optional fields (Python <3.11)
from typing import List
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import StateGraph

from .agents.retriever_agent import RetrieverAgent
from .agents.responder_agent import ResponderAgent


_retriever = RetrieverAgent()


# ---------------------------------------------------------------------------
# LangGraph requires a state schema. We define it as a ``TypedDict`` where
# only ``user_id`` and ``query`` are required. The other keys will be added by
# downstream nodes, so they are marked as ``NotRequired``.
# ---------------------------------------------------------------------------


class QAState(TypedDict):
    """Shared state that flows between graph nodes."""

    user_id: str
    query: str
    # Token de autenticación para el servicio de modelos
    token: str
    # Populated by ``RetrieverAgent``
    context: NotRequired[List[str]]
    # Populated by ``ResponderAgent``
    answer: NotRequired[str]


def answer_user_query(user_id: str, query: str, token: str) -> str:
    """Run the multi-agent chain and return the answer string."""
    # Instantiate a fresh responder for each request with the provided token
    _responder = ResponderAgent(token=token)
    
    # Instantiate the graph with the declared state schema
    _graph = StateGraph(QAState)
    _graph.add_node("retrieve", _retriever)
    _graph.add_node("respond", _responder)
    _graph.set_entry_point("retrieve")
    _graph.add_edge("retrieve", "respond")
    _graph.set_finish_point("respond")

    _chain = _graph.compile()
    
    result = _chain.invoke({
        "user_id": user_id,
        "query": query,
        "token": token
    })
    return result["answer"] 