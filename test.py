# test_langgraph.py

from dotenv import load_dotenv
import os
from typing import Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage
from typing_extensions import TypedDict
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from types import SimpleNamespace

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Supabase client (service-role key for writes)
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
assert SUPABASE_URL and SUPABASE_KEY, "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-discover test IDs from upsell_items
# ─────────────────────────────────────────────────────────────────────────────
ups = supabase.table("upsell_items") \
    .select("property_id") \
    .eq("is_active", True) \
    .limit(1) \
    .execute().data
if not ups:
    raise RuntimeError("No active upsell_items found!")
TEST_PROPERTY_ID = ups[0]["property_id"]

res = supabase.table("reservations") \
    .select("id") \
    .eq("property_id", TEST_PROPERTY_ID) \
    .limit(1) \
    .execute().data
if not res:
    raise RuntimeError(f"No reservations for property {TEST_PROPERTY_ID}")
TEST_RESERVATION_ID = res[0]["id"]

gs = supabase.table("guests") \
    .select("id") \
    .eq("reservation_id", TEST_RESERVATION_ID) \
    .limit(1) \
    .execute().data
if not gs:
    raise RuntimeError(f"No guests for reservation {TEST_RESERVATION_ID}")
TEST_GUEST_ID = gs[0]["id"]

# ─────────────────────────────────────────────────────────────────────────────
# Stub LLM for free testing
# ─────────────────────────────────────────────────────────────────────────────
class EchoLLM:
    def with_structured_output(self, schema):
        return self
    def invoke(self, messages):
        last = messages[-1].content
        return SimpleNamespace(content=f"[echo] {last}")

llm = EchoLLM()

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model (MiniLM)
# ─────────────────────────────────────────────────────────────────────────────
faq_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ─────────────────────────────────────────────────────────────────────────────
# Graph State & Schema
# ─────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: Optional[str]
    reservation_id: Optional[str]
    guest_id: Optional[str]
    property_id: Optional[str]

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions interacting with Supabase
# ─────────────────────────────────────────────────────────────────────────────
def direct_faq_match(question: str, property_id: Optional[str] = None):
    q = supabase.table("faq_entries").select("answer").ilike("question", f"%{question}%")
    if property_id:
        q = q.eq("property_id", property_id)
    res = q.limit(1).execute()
    return res.data[0]["answer"] if res.data else None

def vector_faq_search(question: str, property_id: Optional[str] = None):
    q_vec = faq_embedder.encode(question).tolist()
    res = supabase.rpc("match_faq_vec", {
        "q_vec": q_vec,
        "prop_id": property_id,
        "top_n": 1
    }).execute()
    return res.data[0]["content"] if res.data else None

def fetch_upsell_items(property_id: str, max_price: Optional[int] = None):
    q = (
        supabase.table("upsell_items")
        .select("id, name, description, price")
        .eq("property_id", property_id)
        .eq("is_active", True)
    )
    if max_price is not None:
        q = q.lte("price", max_price)
    return q.order("price").limit(3).execute().data or []

def log_upsell_reco(reservation_id: str, guest_id: str, item_id: str):
    supabase.table("upsell_recommendations").insert({
        "reservation_id": reservation_id,
        "guest_id": guest_id,
        "upsell_item_id": item_id,
        "suggested_by": "ai_agent",
        "status": "suggested",
        "metadata": {},
    }).execute()

# ─────────────────────────────────────────────────────────────────────────────
# Persist + Memorize helper
# ─────────────────────────────────────────────────────────────────────────────
def persist_and_memorize(
    conversation_id: str,
    author_id: Optional[str],
    author_type: str,   # "guest" or "assistant"
    guest_id: str,
    body: str
) -> str:
    # 1️⃣ insert raw message (no .select)
    res = supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "author_id":       author_id,
        "author_type":     author_type,
        "body":            body,
        "guest_id":        guest_id,
    }).execute()
    msg_id = res.data[0]["id"]

    # 2️⃣ embed content
    vec = faq_embedder.encode(body).tolist()

    # 3️⃣ insert into AI memory
    supabase.table("ai_conversation_memory").insert({
        "conversation_id": conversation_id,
        "role":            author_type,
        "content":         body,
        "embedding":       vec,
    }).execute()

    return msg_id

# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def classify_message(state: State):
    last = state["messages"][-1]
    text = last.content.lower()
    if any(w in text for w in ("spa","package","upgrade","add-on")):
        cat = "upsell"
    elif any(w in text for w in ("check-in","check in","time")):
        cat = "booking_faq"
    else:
        cat = "guest_comm"
    return {"category": cat}

def faq_agent(state: State):
    q = state["messages"][-1].content.strip()
    prop = state.get("property_id")
    answer = direct_faq_match(q, prop) or vector_faq_search(q, prop)
    if not answer:
        answer = llm.invoke([
            SystemMessage(content="You are a hotel assistant. Answer factually; say you don't know if unsure."),
            HumanMessage(content=q),
        ]).content
    return {"messages":[{"role":"assistant","content":answer}]}

def upsell_agent(state: State):
    prop   = state["property_id"]
    res_id = state["reservation_id"]
    guest  = state["guest_id"]

    # 1) fetch all recommendations already logged for this reservation
    existing = supabase.table("upsell_recommendations") \
        .select("upsell_item_id") \
        .eq("reservation_id", res_id) \
        .execute().data
    seen_ids = {r["upsell_item_id"] for r in existing}

    # 2) fetch all active upsell items
    items = fetch_upsell_items(prop, None)

    # 3) filter out the ones already suggested
    new_items = [it for it in items if it["id"] not in seen_ids]
    if not new_items:
        return {"messages":[{"role":"assistant","content":"I’m sorry, I don’t have any *new* add-ons right now."}]}

    lines = []
    for it in new_items:
        price_str = f"${it['price']/100:.2f}"
        lines.append(f"• {it['name']} — {price_str}\n  {it['description']}")
        log_upsell_reco(res_id, guest, it["id"])

    response = "Here are a few options you might enjoy:\n" + "\n".join(lines)
    return {"messages":[{"role":"assistant","content":response}]}


def guest_comm_agent(state: State):
    last = state["messages"][-1].content
    reply = llm.invoke([
        SystemMessage(content="You are a friendly hotel guest‑service assistant. Provide concise, polite answers. Escalate to human staff only when necessary."),
        HumanMessage(content=last),
    ]).content
    return {"messages":[{"role":"assistant","content":reply}]}

def router(state: State):
    return {"next": state.get("category","guest_comm")}

# ─────────────────────────────────────────────────────────────────────────────
# Build & compile
# ─────────────────────────────────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("booking_faq", faq_agent)
builder.add_node("upsell", upsell_agent)
builder.add_node("guest_comm", guest_comm_agent)
builder.add_node("router", router)
builder.add_edge(START, "classifier")
builder.add_edge("classifier", "router")
builder.add_conditional_edges(
    "router", lambda s: s.get("next"),
    {"booking_faq":"booking_faq","upsell":"upsell","guest_comm":"guest_comm"}
)
for n in ["booking_faq","upsell","guest_comm"]:
    builder.add_edge(n, END)
graph = builder.compile()

# ─────────────────────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────────────────────
def run_chatbot(res_id: str, guest_id: str, prop_id: str):
    # fetch or create conversation
    resp = supabase.table("conversations").select("id") \
        .eq("reservation_id", res_id).limit(1).execute()
    if resp.data:
        conversation_id = resp.data[0]["id"]
    else:
        conv = supabase.table("conversations").insert({
            "reservation_id": res_id,
            "property_id":    prop_id
        }).execute().data[0]
        conversation_id = conv["id"]

    state = {
        "messages": [], "category": None,
        "reservation_id": res_id,
        "guest_id":       guest_id,
        "property_id":    prop_id,
    }

    print("Type 'exit' to quit.\n")
    while True:
        u = input("Guest > ")
        if u.lower() == "exit":
            break

        persist_and_memorize(
            conversation_id=conversation_id,
            author_id=guest_id,
            author_type="guest",
            guest_id=guest_id,
            body=u
        )

        state["messages"].append({"role":"user","content":u})
        state = graph.invoke(state)
        resp_text = state["messages"][-1].content

        persist_and_memorize(
            conversation_id=conversation_id,
            author_id=None,
            author_type="assistant",
            guest_id=guest_id,
            body=resp_text
        )

        print(f"Assistant > {resp_text}\n")

if __name__ == "__main__":
    run_chatbot(TEST_RESERVATION_ID, TEST_GUEST_ID, TEST_PROPERTY_ID)
