"""Functions for interacting with the Supabase database."""

import logging
from typing import Optional

from postgrest.exceptions import APIError

from client_setup import faq_embedder, supabase

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def direct_faq_match(question: str, property_id: Optional[str] = None):
    """Find a direct FAQ match from the database."""
    try:
        q = supabase.table("faq_entries").select("answer").ilike("question", f"%{question}%")
        if property_id:
            q = q.eq("property_id", property_id)
        res = q.limit(1).execute()
        return res.data[0]["answer"] if res.data else None
    except APIError as e:
        logging.error(f"Database error during direct FAQ match: {e.message}")
        return None

def vector_faq_search(question: str, property_id: Optional[str] = None):
    """Find a FAQ match using vector search."""
    try:
        q_vec = faq_embedder.encode(question).tolist()
        res = supabase.rpc("match_faq_vec", {
            "q_vec": q_vec,
            "prop_id": property_id,
            "top_n": 1
        }).execute()
        return res.data[0]["content"] if res.data else None
    except APIError as e:
        logging.error(f"Database error during vector FAQ search: {e.message}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during vector search: {e}")
        return None

def fetch_upsell_items(property_id: str, max_price: Optional[int] = None):
    """Fetch upsell items for a property."""
    try:
        q = (
            supabase.table("upsell_items")
            .select("id, name, description, price")
            .eq("property_id", property_id)
            .eq("is_active", True)
        )
        if max_price is not None:
            q = q.lte("price", max_price)
        return q.order("price").limit(3).execute().data or []
    except APIError as e:
        logging.error(f"Database error fetching upsell items: {e.message}")
        return []

def log_upsell_reco(reservation_id: str, guest_id: str, item_id: str):
    """Log an upsell recommendation."""
    try:
        supabase.table("upsell_recommendations").insert({
            "reservation_id": reservation_id,
            "guest_id": guest_id,
            "upsell_item_id": item_id,
            "suggested_by": "ai_agent",
            "status": "suggested",
            "metadata": {},
        }).execute()
    except APIError as e:
        logging.error(f"Database error logging upsell recommendation: {e.message}")

def persist_and_memorize(
    conversation_id: str,
    author_id: Optional[str],
    author_type: str,   # "guest" or "assistant"
    guest_id: str,
    body: str
) -> Optional[str]:
    """Persist a message to the database and add it to AI memory."""
    try:
        # 1️⃣ insert raw message
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
    except APIError as e:
        logging.error(f"Database error persisting message: {e.message}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during message persistence: {e}")
        return None 