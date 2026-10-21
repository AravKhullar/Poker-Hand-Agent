"""Evaluation harness built on DeepEval.

Metrics:
  - FaithfulnessMetric   (DeepEval) — are the agent's claims supported by the
                          retrieved context? The standard groundedness metric.
  - AnswerRelevancyMetric (DeepEval) — does the response actually address the
                          spot being asked about?
  - agreement            (custom)   — does the agent's advice match the labeled
                          correct action? Domain-specific; no framework provides
                          this, because the ground truth is poker-specific.

DeepEval defaults to calling OpenAI directly, so we wrap the project's own chat
model (Ollama or Gemini, see llm.py) in DeepEvalBaseLLM to use it as the judge.

Run: uv run python -m pokeragent.eval
"""

import json
import os
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print DeepEval's ⚠ or our ✓.
sys.stdout.reconfigure(encoding="utf-8")

# Must be set before deepeval is imported. A local 9B judge can take minutes
# per metric call, well past DeepEval's default 180s budget.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from pokeragent.agent import build_graph
from pokeragent.llm import CHAT_MODEL, chat_model

EVAL_FILE = "data/eval/hands.jsonl"
HAND_PREFIX = "6-max cash, 100bb effective. "  # shared by every eval hand


class PokerJudge(DeepEvalBaseLLM):
    """Wraps our chat model so DeepEval's metrics use it as the judge
    instead of calling OpenAI directly."""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema:
            return self.model.with_structured_output(schema).invoke(prompt)
        return self.model.invoke(prompt).text

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema:
            return await self.model.with_structured_output(schema).ainvoke(prompt)
        return (await self.model.ainvoke(prompt)).text

    def get_model_name(self) -> str:
        return CHAT_MODEL


_judge = PokerJudge(chat_model)


class EvalCase(BaseModel):
    hand_history: str
    mode: str               # "review" or "simulate"
    correct_action: str     # labeled reference, e.g. "fold", "call", "raise"


def load_eval_set(path: str = EVAL_FILE) -> list[EvalCase]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalCase(**json.loads(line)) for line in lines if line.strip()]


def _to_test_case(graph, case: EvalCase) -> LLMTestCase:
    """Run the agent on one hand and package the result as a DeepEval case."""
    result = graph.invoke({
        "hand_history": case.hand_history,
        "mode": case.mode,
        "steps": 0,
    })
    return LLMTestCase(
        input=case.hand_history,
        actual_output=result["draft"],
        expected_output=case.correct_action,
        retrieval_context=[c.content for c in result["retrieved"]],
    )


# Canonical action for each keyword; aggressive re-raises all count as "raise".
_ACTIONS = {"fold": "fold", "call": "call", "check": "check", "bet": "bet",
            "raise": "raise", "3-bet": "raise", "4-bet": "raise",
            "shove": "raise", "all-in": "raise", "jam": "raise"}
_ACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, _ACTIONS)) + r")\b", re.I)


def first_action(text: str) -> str | None:
    """Earliest poker action named in text, canonicalized."""
    m = _ACTION_RE.search(text)
    return _ACTIONS[m.group(1).lower()] if m else None


def recommended_action(draft: str) -> str | None:
    """The action on the draft's "**Recommendation**:" line."""
    m = re.search(r"Recommendation\W*(.*)", draft, re.I)
    return first_action(m.group(1) if m else draft)


def agrees(case: "EvalCase", tc: LLMTestCase) -> bool:
    """Custom domain metric: does the agent's advice match the labeled action?

    Simulate mode compares actions deterministically — a small judge model
    will happily say "call" matches "raise". Review labels are prose, so those
    go to the judge."""
    if case.mode == "simulate":
        return recommended_action(tc.actual_output) == first_action(case.correct_action)
    prompt = f"""Correct assessment: {tc.expected_output}

Agent's review:
{tc.actual_output}

Does the agent's review reach the same conclusion as the correct assessment? Answer only YES or NO."""
    return _judge.generate(prompt).strip().upper().startswith("YES")


def main() -> None:
    cases = load_eval_set()
    print(f"Running agent ({CHAT_MODEL}) on {len(cases)} eval cases...")
    graph = build_graph()
    test_cases = [_to_test_case(graph, c) for c in cases]

    # DeepEval's standard, research-backed metrics.
    metric_kwargs = {"model": _judge, "threshold": 0.7, "async_mode": False}
    result = evaluate(
        test_cases=test_cases,
        metrics=[FaithfulnessMetric(**metric_kwargs),
                 AnswerRelevancyMetric(**metric_kwargs)],
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        # Sequential: a local Ollama judge gains nothing from concurrent calls.
        async_config=AsyncConfig(run_async=False),
        # DeepEval's disk cache needs pywin32 file locks on Windows and crashes
        # without them; we always want fresh scores anyway.
        cache_config=CacheConfig(write_cache=False),
    )

    # evaluate() doesn't guarantee input order, so key its results by input.
    scores = {
        tr.input: {m.name: m.score for m in tr.metrics_data}
        for tr in result.test_results
    }
    rows, agreed = [], 0
    for case, tc in zip(cases, test_cases, strict=True):
        ok = agrees(case, tc)
        agreed += ok
        s = scores[tc.input]
        said = recommended_action(tc.actual_output) if case.mode == "simulate" else "(review)"
        rows.append(
            f"| {case.hand_history.removeprefix(HAND_PREFIX)} | {case.mode} "
            f"| {case.correct_action} | {said} | {'✓' if ok else '✗'} "
            f"| {s.get('Faithfulness', 0):.2f} "
            f"| {s.get('Answer Relevancy', 0):.2f} |"
        )

    n = len(cases)
    mean = lambda name: sum(s.get(name, 0) for s in scores.values()) / n  # noqa: E731
    print("\n| Hand | Mode | Label | Agrees | Faithfulness | Relevancy |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"\nAgreement with labeled action: {agreed}/{n} ({agreed / n:.0%})")
    print(f"Mean faithfulness: {mean('Faithfulness'):.2f}")
    print(f"Mean answer relevancy: {mean('Answer Relevancy'):.2f}")


if __name__ == "__main__":
    main()
