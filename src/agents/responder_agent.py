"""Responder agent that generates an answer from retrieved context."""
from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, List

from ..config import settings

_SYSTEM_PROMPT = (
    "Eres un asistente experto en información de productos. "
    "Responde utilizando únicamente la información proporcionada en el contexto. "
    "Si no encuentras la respuesta en el contexto, responde con \"Lo siento, no encontré información sobre eso.\""
)


def _build_prompt(context: List[str], question: str) -> str:
    joined_context = "\n---\n".join(context)
    return (
        f"Contexto relevante:\n{joined_context}\n\n"
        f"Pregunta del usuario: {question}\n"
        f"Respuesta del asistente:"
    )


@dataclass
class ResponderAgent:
    """Callable node compatible with LangGraph that adds 'answer' to the state."""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context: List[str] = state.get("context", [])
        question: str = state["query"]
        user_prompt = _build_prompt(context, question)

        payload = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 512,
            "model": settings.MODEL_NAME,
        }

        headers = {
            "Authorization": f"{'PONER CREDENCIALES AQUÍ'}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{settings.ENDPOINT}/chat/completions",
                headers=headers,
                json=payload,
                timeout=settings.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            answer = f"Error al generar la respuesta: {exc}"

        state["answer"] = answer
        return state 