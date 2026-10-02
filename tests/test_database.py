from sqlalchemy import text

from app.db.database import SessionLocal


def test_database_connects_and_has_house_data():
    db = SessionLocal()
    try:
        tables = db.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
        ).scalars().all()
        assert "house_prices" in tables
        assert "conversations" in tables
        assert "messages" in tables
        assert "long_term_memories" in tables
        count = db.execute(text("SELECT COUNT(*) FROM house_prices")).scalar_one()
        assert count >= 300
    finally:
        db.close()
