"""LangGraph agent — the reasoning core."""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pokeragent.llm import chat_model as model
from pokeragent.retrieval import dense_search
from pokeragent.tools import hand_equity

MAX_STEPS = 2

class Action(BaseModel):
    player: str
    action: str
    amount: str = ""  # empty for checks/folds — models often omit it

class StreetActions(BaseModel):
    board: str = Field(description="Cards revealed on this street, e.g. 'Kh 7d 2c'. Empty string for preflop.")
    actions: list[Action] = Field(description="Actions taken this street.")

class ParsedHand(BaseModel):
    hero_position: str = Field(description="Hero's seat relative to the button, e.g. 'BTN', 'SB', 'CO'.")
    stakes: str = Field(description="Game context, e.g. '6-max cash, 100bb effective' or 'MTT, 40bb, 12.5% ante'.")
    hero_cards: str = Field(description="Hero's two hole cards, e.g. 'AhKs'.")
    streets: list[StreetActions] = Field(description="One entry per betting round reached, in order: preflop, flop, turn, river.")
    facing_action: str = Field(description="What Hero must respond to right now, e.g. 'BB checks'. Empty string if the hand is complete (review mode).")

class Critique(BaseModel):
    grounded: bool = Field(description="True if every claim in the draft is supported by the retrieved context.")
    reason: str = Field(description="Brief explanation, especially if the draft is not grounded.")

def full_board(hand: dict) -> str:
    """All community cards dealt so far — each street only lists its own."""
    return " ".join(s["board"] for s in hand["streets"] if s["board"].strip())


class AgentState(TypedDict):
    hand_history: str          # raw input
    mode: str                  # "review" or "simulate"
    parsed_actions: list       # structured decisions to review
    retrieved: list            # RetrievedChunk items for the current query
    equity: dict               # results from the equity tool
    draft: str                 # current analysis draft
    critique: str              # output of the self-critique node
    steps: Annotated[int, lambda a, b: a + b]  # incremented each loop


def parse_hand(state: AgentState) -> dict:
    """Turn a raw hand history into structured actions to review.
    """
    prompt = f"""Extract the following hand history into JSON matching this
shape exactly:
{{"hero_position": str, "stakes": str, "hero_cards": str,
  "streets": [
    {{"board": str, "actions": [{{"player": str, "action": str, "amount": str}}]}}
  ],
  "facing_action": str}}
 
"streets" is one entry per betting round reached (preflop, flop, turn,
river), in order. "board" is the cards revealed on that street ("" for
preflop). Only record actions the hand history explicitly states — never
invent one. "Folds to Hero" means every player before Hero folded; Hero
has NOT acted, so it adds no Hero action.

"facing_action": ONLY if the hand history ends with Hero still to act
(it ends "Hero to act"), describe what Hero is responding to, copied from
THIS hand — the last action by another player (e.g. a raise or a check),
or "folds to Hero" if everyone before Hero folded preflop. Otherwise the
hand is complete and "facing_action" MUST be "".
 
Hand history:
{state["hand_history"]}
 
Respond with ONLY the JSON object, no other text."""

    structured_model = model.with_structured_output(ParsedHand, method="json_schema")
    hand = structured_model.invoke(prompt).model_dump()
    # Review mode means the hand is over; don't trust smaller models to infer
    # that from the text — they often echo the last action as facing_action.
    if state.get("mode") == "review":
        hand["facing_action"] = ""
    else:
        # Simulate mode means Hero is to act, so a trailing Hero action is one
        # the model invented (e.g. reading "Folds to Hero" as Hero folding).
        actions = hand["streets"][-1]["actions"] if hand["streets"] else []
        while actions and actions[-1]["player"].strip().lower() == "hero":
            actions.pop()
    return {"parsed_actions": [hand]}


def retrieve(state: AgentState) -> dict:
    """Pull relevant strategy concepts for the decision under review."""
    hand = state["parsed_actions"][-1]
    query = (
        f"{hand['hero_position']} {hand['stakes']} "
        f"facing {hand.get('facing_action') or 'a bet'} "
        f"on board {full_board(hand) or 'preflop'} "
        f"with hole cards {hand['hero_cards']}"
    )
    results = dense_search(query)
    return {"retrieved": results}


def call_tools(state: AgentState) -> dict:
    """Run the equity calculator on the current spot.
    """
    hand = state["parsed_actions"][-1]

    # ponytail: villain range is fixed to any-two-cards; narrowing it from the
    # action (e.g. a 3-bet range) would make the equity number much sharper.
    equity = hand_equity(
        hero_cards=hand["hero_cards"],
        board=full_board(hand),
        villain_range="any two cards",
    )
    return {"equity": equity.model_dump()}


def generate(state: AgentState) -> dict:
    """Draft the decision-by-decision review from retrieved context + equity.
    """
    hand = state["parsed_actions"][-1]
    context = "\n\n".join(c.content for c in state["retrieved"])
    equity = state["equity"]

    if state["mode"] == "review":
        instruction = (
            "Give a concise decision-by-decision review. For each of Hero's "
            "decisions, output exactly:\n"
            "- **[street] [action]**: ✓ correct / ✗ mistake / ~ marginal — "
            "one sentence why.\n"
            "Keep the whole response under 120 words. End with a one-line "
            "takeaway. Lead with the verdict, not the reasoning."
        )
    else:  # "simulate"
        instruction = (
            "Recommend Hero's best action in under 60 words. Format:\n"
            "- **Recommendation**: [fold/call/raise to X]\n"
            "- **Why**: one or two sentences, citing the key concept and "
            "equity.\n"
            "Lead with the recommendation."
        )

    prompt = f"""{instruction}

Hand history (as written):
{state["hand_history"]}

Parsed:
Position: {hand["hero_position"]}
Stakes: {hand["stakes"]}
Hero cards: {hand["hero_cards"]}
Streets: {hand["streets"]}
Facing: {hand.get("facing_action") or "N/A"}

Hero's equity vs. any two cards: {equity.get("hero_equity", 0):.1%}

Relevant strategy concepts:
{context}"""

    response = model.invoke(prompt)
    # Gemini returns .content as a list of content blocks; .text flattens them
    # to a plain string (and is a no-op for Ollama's plain-string content).
    return {"draft": response.text}

def critique(state: AgentState) -> dict:
    """Check each claim in the draft against the retrieved context.
    """
    context = "\n\n".join(c.content for c in state["retrieved"])
    prompt = f"""Review this poker analysis for claims that contradict or are
unsupported by the retrieved strategy context. A claim is acceptable if it's
consistent with the context and sound poker reasoning — it does NOT need to
be verbatim from the context. Only flag claims that are clearly wrong or
directly contradicted.
Respond with ONLY a JSON object matching: {{"grounded": bool, "reason": str}}

Draft:
{state["draft"]}

Retrieved context:
{context}"""

    structured_model = model.with_structured_output(Critique, method="json_schema")
    response = structured_model.invoke(prompt)
    critique_note = "grounded" if response.grounded else f"ungrounded: {response.reason}"
    return {"critique": critique_note, "steps": 1}

def route_after_critique(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
        return "end"
    if state["critique"].startswith("ungrounded"):
        return "retry"
    return "end"

def build_graph():
    """Wire the nodes into a StateGraph and compile it.
    """
    graph = StateGraph(AgentState)
    graph.add_node(parse_hand)
    graph.add_node(retrieve)
    graph.add_node(call_tools)
    graph.add_node(generate)
    graph.add_node(critique)

    graph.add_edge(START, 'parse_hand')
    graph.add_edge('parse_hand', 'retrieve')
    graph.add_edge('retrieve', 'call_tools')
    graph.add_edge('call_tools', 'generate')
    graph.add_edge('generate', 'critique')

    graph.add_conditional_edges('critique', route_after_critique, {"retry": "generate", "end": END})
    return graph.compile()