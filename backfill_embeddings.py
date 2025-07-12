#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── Init ───
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# ─── Back-fill messages ───
msg_rows = (
    sb.table("messages")
      .select("id, body")
      .is_("embedding", "null")
      .execute()
      .data
)
print(f"Embedding {len(msg_rows)} messages…")
for row in tqdm(msg_rows, desc="Messages"):
    text = row.get("body") or ""
    vec  = model.encode(text).tolist()
    sb.table("messages") \
      .update({"embedding": vec}) \
      .eq("id", row["id"]) \
      .execute()

# ─── Back-fill upsell_items ───
upsell_rows = (
    sb.table("upsell_items")
      .select("id, name, description")
      .is_("embedding", "null")
      .execute()
      .data
)
print(f"Embedding {len(upsell_rows)} upsell_items…")
for row in tqdm(upsell_rows, desc="Upsell Items"):
    text = f"{row.get('name','')} {row.get('description','')}".strip()
    vec  = model.encode(text).tolist()
    sb.table("upsell_items") \
      .update({"embedding": vec}) \
      .eq("id", row["id"]) \
      .execute()

print("✅ Done back-filling messages and upsell_items.")
