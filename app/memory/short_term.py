from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Conversation, Message


def load_recent_messages(conversation_id: str, limit: int = 20) -> list[dict]:
    db: Session = SessionLocal()
    try:
        rows = db.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.desc()).limit(limit)
        ).scalars().all()
        return [{"role": row.role, "content": row.content} for row in reversed(rows)]
    finally:
        db.close()


def save_message(conversation_id: str, role: str, content: str) -> None:
    db: Session = SessionLocal()
    try:
        db.add(Message(conversation_id=conversation_id, role=role, content=content))
        db.commit()
    finally:
        db.close()


def ensure_conversation(user_id: str, conversation_id: str) -> Conversation:
    db: Session = SessionLocal()
    try:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            conversation = Conversation(id=conversation_id, user_id=user_id)
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        return conversation
    finally:
        db.close()
