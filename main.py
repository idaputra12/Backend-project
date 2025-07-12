from dotenv import load_dotenv
import os
from typing import Annotated, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from supabase import create_client, Client

"""
Booking Assistant LangGraph
--------------------------
Implements three specialist skills:
* booking_faq      – first tries an exact / fuzzy match in faq_entries, then vector RAG via ai_documents.
* upsell           – pulls active upsell_items for the guest's property and logs each suggestion in upsell_recommendations.
* guest_comm       – fallback general‑purpose conversational agent.

Assumptions / deps:
* Supabase env vars SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY).
* RPC function match_faq(query text, prop_id uuid, top_n int) RETURNS TABLE(content text, similarity float)
  that performs pgvector similarity search on ai_documents restricted to the property.
* pgcrypto extension enabled for UUID generation in the DB (already created in migration script).
"""

load_dotenv()

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
assert SUPABASE_URL and SUPABASE_KEY, "Set SUPABASE_URL and SUPABASE_SERVICE_KEY/ANON_KEY in env"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# ---------------------------------------------------------------------------
# Graph State & Schema
# ---------------------------------------------------------------------------
class MessageClassifier(BaseModel):
    """Lightweight classifier guiding the router node."""

    category: Literal["booking_faq", "upsell", "guest_comm"] = Field(
        ..., description="Which specialist agent should answer this message?"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: str | None
    reservation_id: str | None
    guest_id: str | None
    property_id: str | None  # used for tenant scoping


# ---------------------------------------------------------------------------
# Helper functions interacting with Supabase
# ---------------------------------------------------------------------------
def direct_faq_match(question: str, property_id: Optional[str] = None):
    """Exact / fuzzy match (ILIKE) in faq_entries."""
    query = supabase.table("faq_entries").select("answer").ilike("question", f"%{question}%")
    if property_id:
        query = query.eq("property_id", property_id)
    res = query.limit(1).execute()
    return res.data[0]["answer"] if res.data else None


def vector_faq_search(question: str, property_id: Optional[str] = None):
    """Vector similarity search via match_faq RPC. Returns best content string or None."""
    params = {"query": question, "prop_id": property_id, "top_n": 1}
    res = supabase.rpc("match_faq", params).execute()
    return res.data[0]["content"] if res.data else None


def fetch_upsell_items(property_id: str, max_price_cents: Optional[int] = None):
    query = (
        supabase.table("upsell_items")
        .select("id, name, description, price_cents")
        .eq("property_id", property_id)
        .eq("is_active", True)
    )
    if max_price_cents is not None:
        query = query.lte("price_cents", max_price_cents)
    result = query.order("price_cents").limit(3).execute()
    return result.data or []


def log_upsell_reco(reservation_id: str, guest_id: str, item_id: str):
    supabase.table("upsell_recommendations").insert(
        {
            "reservation_id": reservation_id,
            "guest_id": guest_id,
            "upsell_item_id": item_id,
            "suggested_by": "ai_agent",
            "status": "suggested",
            "metadata": {},
        }
    ).execute()


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def classify_message(state: State):
    last_msg = state["messages"][-1]
    cls_llm = llm.with_structured_output(MessageClassifier)
    result = cls_llm.invoke(
        [
            SystemMessage(
                content="""Classify the user message into one of: \n- booking_faq: questions about hotel policies, check‑in/out, amenities, parking, pets, etc.\n- upsell: user wants upgrades, add‑ons, spa, late checkout, or the assistant could proactively recommend.\n- guest_comm: everything else (requests, complaints, small‑talk)."""
            ),
            HumanMessage(content=last_msg.content),
        ]
    )
    return {"category": result.category}


def faq_agent(state: State):
    question = state["messages"][-1].content.strip()
    property_id = state.get("property_id")

    # 1) direct match in faq_entries
    answer = direct_faq_match(question, property_id)
    if not answer:
        # 2) vector search fallback
        answer = vector_faq_search(question, property_id)
    if not answer:
        # 3) let the LLM handle it with a policy not to hallucinate
        messages = [
            SystemMessage(content="You are a hotel assistant. Answer factually; say you don't know if unsure."),
            HumanMessage(content=question),
        ]
        answer = llm.invoke(messages).content
    return {"messages": [{"role": "assistant", "content": answer}]}


def upsell_agent(state: State):
    property_id = state.get("property_id")
    reservation_id = state.get("reservation_id")
    guest_id = state.get("guest_id")
    last_msg = state["messages"][-1]

    # Simple heuristic: if user provided a budget, you could parse it here
    budget_cents: Optional[int] = None

    items = fetch_upsell_items(property_id, budget_cents)
    if not items:
        content = "I’m sorry, I don’t have any special add‑ons available right now."
        return {"messages": [{"role": "assistant", "content": content}]}

    # Build suggestions and log them
    lines = []
    for item in items:
        price = f"${item['price_cents'] / 100:.2f}" if item["price_cents"] else "Priced on request"
        lines.append(f"• {item['name']} — {price}\n  {item['description']}")
        log_upsell_reco(reservation_id, guest_id, item["id"])

    response = "Here are a few options you might enjoy:\n" + "\n".join(lines)
    return {"messages": [{"role": "assistant", "content": response}]}


def guest_comm_agent(state: State):
    last_msg = state["messages"][-1]
    reply = llm.invoke(
        [
            SystemMessage(
                content="You are a friendly hotel guest‑service assistant. Provide concise, polite answers. Escalate to human staff only when necessary."
            ),
            HumanMessage(content=last_msg.content),
        ]
    ).content
    return {"messages": [{"role": "assistant", "content": reply}]}


def router(state: State):
    return {"next": state.get("category", "guest_comm")}


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("booking_faq", faq_agent)
graph_builder.add_node("upsell", upsell_agent)
graph_builder.add_node("guest_comm", guest_comm_agent)
graph_builder.add_node("router", router)

# edges
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda s: s.get("next"),
    {"booking_faq": "booking_faq", "upsell": "upsell", "guest_comm": "guest_comm"},
)

for node in ["booking_faq", "upsell", "guest_comm"]:
    graph_builder.add_edge(node, END)

graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# CLI runner (optional)
# ---------------------------------------------------------------------------

def run_chatbot(reservation_id: str, guest_id: str, property_id: str):
    state: State = {
        "messages": [],
        "category": None,
        "reservation_id": reservation_id,
        "guest_id": guest_id,
        "property_id": property_id,
    }

    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("Guest > ")
        if user_input.lower() == "exit":
            break

        state["messages"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)
        assistant_reply = state["messages"][-1]["content"]
        print(f"Assistant > {assistant_reply}\n")


if __name__ == "__main__":
    run_chatbot(
        reservation_id=os.getenv("TEST_RESERVATION_ID", "00000000-0000-0000-0000-000000000000"),
        guest_id=os.getenv("TEST_GUEST_ID", "00000000-0000-0000-0000-000000000000"),
        property_id=os.getenv("TEST_PROPERTY_ID", "00000000-0000-0000-0000-000000000000"),
    )
