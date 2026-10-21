"""Model config — the one place that decides which provider serves chat and
embeddings. Both are LangChain "provider:model" strings, so switching between
local Ollama and the Gemini API is an .env change, not a code change:

  CHAT_MODEL=ollama:qwen3.5:9b                 (local, no quota)
  CHAT_MODEL=google_genai:gemini-3.8-flash     (needs GOOGLE_API_KEY)
  EMBEDDING_MODEL=ollama:nomic-embed-text      (768 dims — matches schema.sql)
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "ollama:qwen3.5:9b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")

if CHAT_MODEL.startswith("ollama:"):
    # reasoning=False turns off qwen-style <think> output, which is slow and
    # leaks into the draft; num_ctx fits ~10 retrieved chunks plus the hand.
    _chat_kwargs = {"reasoning": False, "num_ctx": 8192}
else:
    _chat_kwargs = {"max_retries": 5}

chat_model = init_chat_model(CHAT_MODEL, temperature=0, **_chat_kwargs)
embeddings = init_embeddings(EMBEDDING_MODEL)
