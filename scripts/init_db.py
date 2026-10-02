import os
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db.database import SessionLocal, create_all_tables


def apply_schema() -> None:
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    db = SessionLocal()
    try:
        db.execute(text(sql))
        db.commit()
    finally:
        db.close()


def reset_seed_data() -> None:
    db = SessionLocal()
    try:
        db.execute(text("TRUNCATE TABLE messages RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE conversations RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE long_term_memories RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE house_prices RESTART IDENTITY CASCADE;"))
        db.commit()
    finally:
        db.close()


def insert_seed_data() -> int:
    from db.seed import seed_database

    return seed_database()


if __name__ == "__main__":
    get_settings()
    create_all_tables()
    apply_schema()
    reset_seed_data()
    count = insert_seed_data()
    print(f"Database ready. House records: {count}")
