"""Evaluation harness — LLM-as-judge + agreement with labeled hands.

TODO(you): The third file you implement yourself. This is your differentiator
— most portfolio RAG projects have no eval at all, and you've built one before
at work, so lean into it.

Two things to measure:

  1. Agreement with reference actions. Load a labeled set of hands (see
     data/eval and docs/EVAL.md) where each decision has a reference "correct"
     action derived from a solver or established charts. Run the agent, compare
     its recommendation to the reference, report agreement %.

  2. Groundedness (LLM-as-judge). For each review the agent produces, prompt a
     judge model to score whether each claim is supported by the retrieved
     context and whether the advice is actionable. Report the average, and
     crucially report it BEFORE vs AFTER enabling the self-critique node so you
     can show the loop helped.

Keep the judge prompt and rubric in version control. Attach an eval result to
changes that touch retrieval or the agent, the way you would in production.
"""

from dataclasses import dataclass


@dataclass
class EvalCase:
    hand_history: str
    reference_action: str


def load_eval_set(path: str) -> list[EvalCase]:
    """Load labeled hands from disk.

    TODO(you): define the file format and parse it.
    """
    raise NotImplementedError


def score_agreement(cases: list[EvalCase]) -> float:
    """Fraction of decisions where the agent matches the reference action.

    TODO(you): run the agent per case and compare.
    """
    raise NotImplementedError


def score_groundedness(cases: list[EvalCase]) -> float:
    """LLM-as-judge groundedness/actionability score.

    TODO(you): prompt a judge model with the review + retrieved context.
    """
    raise NotImplementedError