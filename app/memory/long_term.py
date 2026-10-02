from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import LongTermMemory


def load_user_memories(user_id: str) -> list[dict]:
    db: Session = SessionLocal()
    try:
        rows = db.execute(select(LongTermMemory).where(LongTermMemory.user_id == user_id).order_by(LongTermMemory.created_at.desc())).scalars().all()
        return [
            {"id": row.id, "memory_type": row.memory_type, "content": row.content, "metadata": row.metadata or {}} for row in rows
        ]
    finally:
        db.close()


def save_user_memory(user_id: str, memory_type: str, content: str, metadata: dict | None = None) -> LongTermMemory:
    db: Session = SessionLocal()
    try:
        entry = LongTermMemory(user_id=user_id, memory_type=memory_type, content=content, metadata=metadata or {})
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    finally:
        db.close()
