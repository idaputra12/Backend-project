"""Initialize and configure external clients."""

import os
from types import SimpleNamespace

from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

from config import EMBEDDING_MODEL

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Supabase client (service-role key for writes)
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Stub LLM for free testing
# ─────────────────────────────────────────────────────────────────────────────
class EchoLLM:
    """A stub LLM that echoes the last message or returns a structured response."""
    def __init__(self):
        self._structured_output_schema = None

    def with_structured_output(self, schema):
        self._structured_output_schema = schema
        return self

    def invoke(self, messages):
        last_content = messages[-1].content.lower()

        if self._structured_output_schema:
            # Specific logic for Category classification
            if self._structured_output_schema.__name__ == 'Category':
                if any(w in last_content for w in ("spa", "package", "upgrade", "add-on")):
                    cat = "upsell"
                elif any(w in last_content for w in ("checkin", "check-in", "check in", "time")):
                    cat = "booking_faq"
                else:
                    cat = "guest_comm"
                
                # Create the structured output object BEFORE resetting the schema
                result = self._structured_output_schema(name=cat)
                self._structured_output_schema = None
                return result
            
            # Reset for the next call in case it's not the Category schema
            self._structured_output_schema = None

        return SimpleNamespace(content=f"[echo] {last_content}")

llm = EchoLLM()

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model (MiniLM)
# ─────────────────────────────────────────────────────────────────────────────
faq_embedder = SentenceTransformer(EMBEDDING_MODEL) 