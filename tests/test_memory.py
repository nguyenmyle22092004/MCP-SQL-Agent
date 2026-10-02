from app.memory.long_term import load_user_memories, save_user_memory
from app.memory.short_term import ensure_conversation, load_recent_messages, save_message


def test_short_term_memory_round_trip():
    conversation_id = "mem-conv-1"
    user_id = "user-memory-test"
    ensure_conversation(user_id, conversation_id)
    save_message(conversation_id, "user", "How many 3-bedroom houses are in Boston?")
    save_message(conversation_id, "assistant", "42")
    messages = load_recent_messages(conversation_id, limit=10)
    assert any(m["content"] == "How many 3-bedroom houses are in Boston?" for m in messages)
    assert any(m["content"] == "42" for m in messages)


def test_long_term_memory_round_trip():
    user_id = "user-long-term"
    save_user_memory(user_id, "preference", "User is mainly interested in Boston houses.", metadata={"city": "Boston"})
    memories = load_user_memories(user_id)
    assert any(m["memory_type"] == "preference" for m in memories)
    assert any("Boston" in m["content"] for m in memories)
