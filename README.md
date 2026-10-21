# Poker Hand Review Agent

An agentic assistant that reviews No-Limit Hold'em hands and recommends
in-the-moment decisions, grounding every claim in a retrieved strategy
concept plus a real equity calculation.

![Demo UI](docs/demo.png)

## Problem

Reviewing your own poker hands to find mistakes is slow and most players do
it badly. There's no fast, objective way to check a decision — you'd have to
remember the relevant strategy concept yourself, or manually run an equity
calculator, or dig through articles. Most players don't bother, so they keep
repeating the same leaks.

## Solution

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that:

1. Parses a hand history into structured, street-by-street actions.
2. Retrieves relevant strategy concepts from a corpus embedded in Postgres +
   pgvector.
3. Calls a real equity-calculation tool ([PokerKit](https://pypi.org/project/pokerkit/))
   on the full board to ground claims in actual numbers, not guesses.
4. Drafts an analysis, then runs a self-critique pass that checks every
   claim against the retrieved context and flags anything unsupported.

The agent runs in two modes:

- **Review mode** — evaluate whether Hero's actual decisions in a completed
  hand were correct.
- **Simulate mode** — given a hand history that stops mid-decision,
  recommend what Hero should do right now.

It's served by FastAPI, with a one-page demo UI at `/`.

## Result

RESULTS_PLACEHOLDER

## Architecture

```
Hand history
     |
     v
[parse_hand] -> [retrieve] -> [call_tools] -> [generate]
                                                  ^   |
                                                  |   v
                                   retry (ungrounded) [critique] -> (grounded) -> END
```

Data lives in two tables — `documents` (one row per ingested file) and
`chunks` (retrievable unit, 768-dim embedding stored on the row, HNSW-indexed).
A `MAX_STEPS` cap on the critique loop prevents infinite retries.

### Model providers

Chat and embeddings are LangChain `provider:model` strings in `.env`
(see [`llm.py`](src/pokeragent/llm.py)), so switching providers is a config
change, not a code change:

| Setting | Local (default) | Hosted |
|---|---|---|
| `CHAT_MODEL` | `ollama:qwen3.5:9b` — no quota, no key | `google_genai:gemini-3.8-flash` + `GOOGLE_API_KEY` (free tier ≈ 20 req/day ≈ 5 hands) |
| `EMBEDDING_MODEL` | `ollama:nomic-embed-text` (768 dims) | — |

Structured outputs (hand parsing, critique) use JSON-schema constrained
decoding rather than tool calls; small local models emit malformed tool calls
often enough to crash runs, while constrained decoding can't produce invalid
JSON. Changing the embedding model means changing `vector(768)` in
`schema.sql` and re-running ingestion.

## Tech stack

Python 3.12, LangGraph, LangChain, Ollama / Gemini, Postgres + pgvector,
FastAPI, PokerKit (equity), DeepEval, pydantic, pytest, ruff, GitHub Actions.

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Pull local models (skip the chat model if using Gemini)
ollama pull nomic-embed-text
ollama pull qwen3.5:9b

# 3. Start Postgres + pgvector (schema.sql runs on first start)
docker compose up -d

# 4. Configure
cp .env.example .env      # pick CHAT_MODEL; add GOOGLE_API_KEY for Gemini

# 5. Build the strategy corpus, then embed it
uv run --group scripts python scripts/fetch_wiki.py
uv run --group scripts python scripts/fetch_other.py
uv run python -m pokeragent.chunk_docs

# 6. Run the API + demo UI
uv run uvicorn pokeragent.main:app --reload
```

## Demo

Open <http://localhost:8000> — paste a hand (or click an example), pick
**Review** or **Simulate**, and hit **Analyze**. Or call the API directly:

```bash
curl -X POST localhost:8000/review -H "Content-Type: application/json" -d '{
  "hand_history": "6-max cash, 100bb effective. Hero on BTN with AhKs. Hero raises to 2.5bb, SB calls. Flop: Kh 7d 2c. SB checks, Hero bets 3bb, SB calls. Turn: 9s. SB checks, Hero bets 8bb, SB folds.",
  "mode": "review"
}'
```

DEMO_PLACEHOLDER

## Testing

```bash
# fast unit tests only — no DB, no model server needed (this is what CI runs)
uv run pytest -m "not integration" -q

# full suite — needs Postgres, Ollama/Gemini and an ingested corpus
uv run pytest -v

# labeled-hand evaluation (DeepEval faithfulness/relevancy + action agreement)
uv run python -m pokeragent.eval
```

TESTS_PLACEHOLDER

## Project status

Working end to end: ingestion, dense vector retrieval, equity calculation,
the full LangGraph pipeline (both modes), the API + demo UI, and the
evaluation harness.

Open:
- `hybrid_search` (dense + full-text, RRF) is implemented in `retrieval.py`
  but the agent uses dense-only; switch once the eval shows dense missing
  exact poker terminology.
- Villain range for equity is fixed to any-two-cards; narrowing it from the
  action (e.g. a 3-bet range) would sharpen the numbers.
- The eval set is 10 hands — enough to catch regressions, too small for
  strong claims.
- Deployment — not yet live.

## License

MIT
