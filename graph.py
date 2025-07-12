"""LangGraph definition and nodes."""

from typing import Annotated, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from client_setup import llm
from database_ops import (direct_faq_match, fetch_upsell_items,
                          log_upsell_reco, vector_faq_search)


# ─────────────────────────────────────────────────────────────────────────────
# Graph State & Schema
# ─────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: Optional[str]
    reservation_id: Optional[str]
    guest_id: Optional[str]
    property_id: Optional[str]

class Category(BaseModel):
    """The category of the user's message."""
    name: Literal["booking_faq", "upsell", "guest_comm"]

# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────
def classify_message(state: State):
    """Classify the user's message using the LLM."""
    last_message = state["messages"][-1]
    classifying_llm = llm.with_structured_output(Category)
    
    result = classifying_llm.invoke([
        SystemMessage(content="Classify the user's message into one of the following categories: booking_faq, upsell, or guest_comm."),
        HumanMessage(content=last_message.content)
    ])
    
    return {"category": result.name}


def faq_agent(state: State):
    """Handle booking-related frequently asked questions."""
    q = state["messages"][-1].content.strip()
    prop = state.get("property_id")
    answer = direct_faq_match(q, prop) or vector_faq_search(q, prop)
    if not answer:
        answer = llm.invoke([
            SystemMessage(content="You are a hotel assistant. Answer factually; say you don't know if unsure."),
            HumanMessage(content=q),
        ]).content
    return {"messages": [{"role": "assistant", "content": answer}]}


def upsell_agent(state: State):
    """Handle upselling of products and services."""
    prop = state["property_id"]
    res_id = state["reservation_id"]
    guest = state["guest_id"]

    # This logic is simplified; in a real scenario, you'd check existing recommendations.
    items = fetch_upsell_items(prop, None)
    if not items:
        return {"messages": [{"role": "assistant", "content": "I'm sorry, I don't have any add-ons right now."}]}

    lines = []
    for it in items:
        price_str = f"${it['price']/100:.2f}"
        lines.append(f"• {it['name']} — {price_str}\n  {it['description']}")
        log_upsell_reco(res_id, guest, it["id"])

    response = "Here are a few options you might enjoy:\n" + "\n".join(lines)
    return {"messages": [{"role": "assistant", "content": response}]}


def guest_comm_agent(state: State):
    """Handle general guest communication."""
    last = state["messages"][-1].content
    reply = llm.invoke([
        SystemMessage(content="You are a friendly hotel guest-service assistant. Provide concise, polite answers. Escalate to human staff only when necessary."),
        HumanMessage(content=last),
    ]).content
    return {"messages": [{"role": "assistant", "content": reply}]}


def router(state: State):
    """Route to the appropriate agent based on the category."""
    return state.get("category", "guest_comm")

# ─────────────────────────────────────────────────────────────────────────────
# Build & compile graph
# ─────────────────────────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("booking_faq", faq_agent)
builder.add_node("upsell", upsell_agent)
builder.add_node("guest_comm", guest_comm_agent)

builder.add_conditional_edges(
    "classifier",
    router,
    {"booking_faq": "booking_faq", "upsell": "upsell", "guest_comm": "guest_comm"}
)

builder.add_edge(START, "classifier")
for n in ["booking_faq", "upsell", "guest_comm"]:
    builder.add_edge(n, END)

graph = builder.compile() 