from __future__ import annotations

import csv
import random
from pathlib import Path

from sqlalchemy import text

from app.config import get_settings
from app.db.database import SessionLocal

CITIES = {
    "Boston": "MA",
    "Seattle": "WA",
    "Austin": "TX",
    "Chicago": "IL",
    "Denver": "CO",
    "San Francisco": "CA",
    "New York": "NY",
    "Portland": "OR",
    "Atlanta": "GA",
    "Dallas": "TX",
}


def generate_house_records(count: int = 360) -> list[dict]:
    rng = random.Random(42)
    records: list[dict] = []
    city_order = list(CITIES.keys())
    for i in range(count):
        city = city_order[i % len(city_order)]
        state = CITIES[city]
        bedrooms = rng.randint(1, 6)
        bathrooms = round(rng.uniform(1.0, 5.0), 1)
        area = rng.randint(500, 5000)
        year_built = rng.randint(1950, 2025)
        city_multiplier = {
            "Boston": 1.25,
            "Seattle": 1.18,
            "Austin": 1.08,
            "Chicago": 1.0,
            "Denver": 1.07,
            "San Francisco": 1.4,
            "New York": 1.45,
            "Portland": 1.06,
            "Atlanta": 0.92,
            "Dallas": 0.9,
        }[city]
        base = 120000 + bedrooms * 25000 + (area * 120) + (year_built - 1950) * 1500
        price = base * city_multiplier * (0.85 + rng.random() * 0.55)
        price = round(price, 2)
        address = f"{rng.randint(100, 9999)} {['Oak', 'Maple', 'Cedar', 'Elm', 'Pine', 'Willow', 'Sunset', 'River', 'Lake', 'Hill'][rng.randint(0, 9)]} St"
        records.append(
            {
                "address": address,
                "city": city,
                "state": state,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "area_sqft": area,
                "year_built": year_built,
                "price": price,
            }
        )
    return records


def ensure_schema() -> None:
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    text_sql = schema_path.read_text(encoding="utf-8")
    db = SessionLocal()
    try:
        db.execute(text(text_sql))
        db.commit()
    finally:
        db.close()


def seed_database() -> int:
    ensure_schema()
    db = SessionLocal()
    try:
        db.execute(text("TRUNCATE TABLE house_prices RESTART IDENTITY CASCADE;"))
        db.execute(
            text(
                "INSERT INTO house_prices (address, city, state, bedrooms, bathrooms, area_sqft, year_built, price) VALUES (:address, :city, :state, :bedrooms, :bathrooms, :area_sqft, :year_built, :price)"
            ),
            [
                {
                    "address": row["address"],
                    "city": row["city"],
                    "state": row["state"],
                    "bedrooms": row["bedrooms"],
                    "bathrooms": row["bathrooms"],
                    "area_sqft": row["area_sqft"],
                    "year_built": row["year_built"],
                    "price": row["price"],
                }
                for row in generate_house_records(360)
            ],
        )
        db.commit()
        return db.execute(text("SELECT COUNT(*) FROM house_prices")).scalar_one()
    finally:
        db.close()


if __name__ == "__main__":
    count = seed_database()
    print(f"Inserted {count} house records.")
