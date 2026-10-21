"""FastAPI service exposing the review/simulate endpoint, plus a one-page demo
UI at /.

Run: uv run uvicorn pokeragent.main:app --reload
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pokeragent.agent import build_graph

app = FastAPI(title="Poker Hand Review Agent")

# Built once at import time — the graph is just wired-up Python objects,
# no per-request state, so there's no reason to rebuild it on every call.
_graph = build_graph()


class ReviewRequest(BaseModel):
    hand_history: str = Field(min_length=1, max_length=5000)
    mode: Literal["review", "simulate"] = "review"


class ReviewResponse(BaseModel):
    review: str
    grounded: bool
    retries: int


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest) -> ReviewResponse:
    try:
        result = _graph.invoke({
            "hand_history": req.hand_history,
            "mode": req.mode,
            "steps": 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from None

    return ReviewResponse(
        review=result["draft"],
        grounded=result["critique"].startswith("grounded"),
        retries=result["steps"],
    )