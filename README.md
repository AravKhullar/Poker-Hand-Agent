# Poker Hand Review Agent

An agentic assistant that reviews No-Limit Hold'em hand histories, flags
decisions that deviate from sound play, and grounds every flag in a retrieved
strategy concept plus an equity calculation.

<!-- Add a screenshot or a short GIF of the live demo here once deployed. It is
the single highest-leverage thing on this page for a reader skimming in 15
seconds. -->

## Problem

Reviewing your own poker hands to find mistakes is slow and most players do it
badly. Off the table there is no objective, on-demand reference for what the
correct decision was, so players either skip review entirely or rely on vague
recollection of strategy advice. The information needed to judge a decision
(preflop ranges, pot odds, board texture, position) lives scattered across
articles and books, and a single keyword search rarely surfaces the specific
concept a given spot calls for.

## Solution

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that:

1. Parses a hand history into structured actions.
2. Retrieves the relevant strategy concepts from a corpus embedded in
   Postgres + pgvector, using hybrid (BM25 + dense) search with reranking.
3. Calls an equity-calculation tool to ground claims in real numbers.
4. Drafts an analysis, then runs a self-critique pass that checks each claim
   against the retrieved context and re-retrieves if a claim is unsupported.
5. Returns a decision-by-decision review with citations.

The system is wrapped in a FastAPI service, containerized with Docker, and
traced with LangSmith.

## Result

<!-- Fill these in after building the eval. Report the honest measured numbers.
Do NOT invent figures. -->

- Agreement with reference "correct" actions: __% on a __-hand labeled test set
- Groundedness (LLM-as-judge): improved from __ to __ after adding the
  self-critique node
- Median review latency: __ ms
- Corpus size: __ strategy documents / __ retrievable chunks

See [`docs/EVAL.md`](docs/EVAL.md) for how the labeled test set was built and
how "correct action" is defined.

## Architecture

```
Hand history
     |
     v
[parse] -> [retrieve: BM25 + dense + rerank] -> [call equity tool]
     |                                                |
     +-----------------------> [generate] <-----------+
                                    |
                              [self-critique] --unsupported claim--> [retrieve]
                                    |
                                 (grounded)
                                    v
                             Decision review
```

Data lives in three tables (`documents` -> `chunks` -> `embeddings`) so the
corpus can be re-embedded with a new model without rebuilding source records.

## Tech stack

Python 3.12, LangGraph, LangChain, Postgres + pgvector, FastAPI, Docker,
PokerKit (equity), LangSmith (tracing), pytest, ruff, GitHub Actions.

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Start Postgres + pgvector
docker compose up -d

# 3. Apply the schema
psql "$DATABASE_URL" -f schema.sql

# 4. Configure secrets
cp .env.example .env      # then fill in the values

# 5. Ingest the strategy corpus
uv run python -m pokeragent.ingest data/corpus

# 6. Run the API
uv run uvicorn pokeragent.main:app --reload
```

Then POST a hand history to `http://localhost:8000/review`.

## Project status

Work in progress. See the `TODO(you)` markers in `retrieval.py`, `agent.py`,
and `eval.py` for the components under active development.

## License

MIT