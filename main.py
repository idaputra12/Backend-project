"""Main entry point for the chatbot CLI."""

import logging

from client_setup import supabase
from database_ops import persist_and_memorize
from graph import graph
import os
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Test constants ---
TEST_PROPERTY_ID = os.getenv("TEST_PROPERTY_ID")
TEST_RESERVATION_ID = os.getenv("TEST_RESERVATION_ID")
TEST_GUEST_ID = os.getenv("TEST_GUEST_ID")


def run_chatbot(res_id: str, guest_id: str, prop_id: str):
    """Run the chatbot CLI."""
    # Fetch or create conversation
    try:
        resp = supabase.table("conversations").select("id").eq("reservation_id", res_id).limit(1).execute()
        if resp.data:
            conversation_id = resp.data[0]["id"]
        else:
            conv = supabase.table("conversations").insert({
                "reservation_id": res_id,
                "property_id": prop_id
            }).execute().data[0]
            conversation_id = conv["id"]
    except Exception as e:
        logging.error(f"Could not fetch or create conversation: {e}")
        return

    state = {
        "messages": [],
        "reservation_id": res_id,
        "guest_id": guest_id,
        "property_id": prop_id,
    }

    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("Guest > ")
        if user_input.lower() == "exit":
            break

        persist_and_memorize(
            conversation_id=conversation_id,
            author_id=guest_id,
            author_type="guest",
            guest_id=guest_id,
            body=user_input
        )

        state["messages"].append({"role": "user", "content": user_input})
        
        try:
            state = graph.invoke(state)
            response_text = state["messages"][-1].content

            persist_and_memorize(
                conversation_id=conversation_id,
                author_id=None,
                author_type="assistant",
                guest_id=guest_id,
                body=response_text
            )
            print(f"Assistant > {response_text}\n")
        except Exception as e:
            logging.error(f"Error invoking graph: {e}")
            print("Assistant > I'm sorry, I encountered an error. Please try again.\n")


if __name__ == "__main__":
    run_chatbot(TEST_RESERVATION_ID, TEST_GUEST_ID, TEST_PROPERTY_ID)
