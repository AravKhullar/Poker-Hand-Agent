"""LangGraph agent — the reasoning core.
"""

from dataclasses import asdict
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from pokeragent.retrieval import dense_search
from pokeragent.tools import hand_equity

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("CHAT_MODEL"),
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = os.getenv("OPENROUTER_BASE_URL"),
)

MAX_STEPS = 6

class Action(BaseModel):
    player: str
    action: str
    amount: str

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
preflop). If the hand history ends mid-decision (Hero must act next),
set "facing_action" to what Hero is responding to (e.g. "BB checks"),
and don't include Hero's response in "actions" since it hasn't happened
yet. If the hand is complete, set "facing_action" to "".
 
Hand history:
{state["hand_history"]}
 
Respond with ONLY the JSON object, no other text."""

    structured_model = model.with_structured_output(ParsedHand)
    response = structured_model.invoke(prompt)

    return {"parsed_actions": [response.model_dump()]}


def retrieve(state: AgentState) -> dict:
    """Pull relevant strategy concepts for the decision under review."""
    hand = state["parsed_actions"][-1]
    last_street = hand["streets"][-1]
    query = (
        f"{hand['hero_position']} {hand['stakes']} "
        f"facing {hand.get('facing_action') or 'a bet'} "
        f"on board {last_street['board']} "
        f"with hole cards {hand['hero_cards']}"
    )    
    results = dense_search(query)
    return {"retrieved": results}


def call_tools(state: AgentState) -> dict:
    """Run the equity calculator on the current spot.
    """
    hand = state["parsed_actions"][-1]
    last_street = hand["streets"][-1]

    equity = hand_equity(
        hero_cards=hand["hero_cards"],
        board=last_street["board"],
        villain_range="any two cards", #Implement range
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
            "Evaluate whether Hero's actual decisions in this hand were "
            "correct. For each decision, say whether it was right or "
            "wrong and why, citing the strategy concepts below."
        )
    else:  # "simulate"
        instruction = (
            "Hero is facing a decision right now. Recommend the best "
            "action (fold, call, or raise with a sizing), citing the "
            "strategy concepts and equity below."
        )

    prompt = f"""{instruction}
Hand:
Position: {hand["hero_position"]}
Stakes: {hand["stakes"]}
Hero cards: {hand["hero_cards"]}
Streets: {hand["streets"]}
Facing: {hand.get("facing_action") or "N/A"}
 
Hero's equity: {equity.get("hero_equity", 0):.1%} (share of pot on average)
 
Relevant strategy concepts:
{context}"""

    response = model.invoke(prompt)
    return {"draft": response.content}

def critique(state: AgentState) -> dict:
    """Check each claim in the draft against the retrieved context.
    """
    context = "\n\n".join(c.content for c in state["retrieved"])
    prompt = f"""Does every claim in the draft below have support in the
retrieved context? A claim about equity numbers is grounded if it matches
the equity figures given; a strategy claim is grounded if it's backed by
the retrieved concepts.
 
Draft:
{state["draft"]}
 
Retrieved context:
{context}"""
    
    structured_model = model.with_structured_output(Critique)
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

    graph.add_conditional_edges('critique', route_after_critique, {"retry": "retrieve", "end": END})
    return graph.compile()