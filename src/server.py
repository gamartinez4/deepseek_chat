"""FastAPI app exposing the /query endpoint."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import answer_user_query

app = FastAPI(title="Product Query RAG Bot")


class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    try:
        answer = answer_user_query(body.user_id, body.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(answer=answer) 