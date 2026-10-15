from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from app.config import get_readonly_database_url, get_settings


def _default_cache_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "gazetteer.json"


def build_gazetteer_from_database(cache_path: str | Path | None = None) -> dict[str, Any]:
    """Build and cache the entity dictionary only through a DB role without write grants."""
    settings = get_settings()
    engine = create_engine(get_readonly_database_url(settings), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                permissions = connection.execute(text(
                    "SELECT current_user, "
                    "has_table_privilege(current_user, 'public.house_prices', 'SELECT'), "
                    "has_table_privilege(current_user, 'public.house_prices', 'INSERT'), "
                    "has_table_privilege(current_user, 'public.house_prices', 'UPDATE'), "
                    "has_table_privilege(current_user, 'public.house_prices', 'DELETE'), "
                    "has_table_privilege(current_user, 'public.house_prices', 'TRUNCATE')"
                )).one()
                role, can_select, can_insert, can_update, can_delete, can_truncate = permissions
                if not can_select or can_insert or can_update or can_delete or can_truncate:
                    raise PermissionError(
                        f"Gazetteer requires a read-only DB role; current_user={role!r} has unsafe table privileges."
                    )
                columns = connection.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'house_prices'"
                )).scalars().all()
                required = {"address", "city", "state"}
                missing = required.difference(columns)
                if missing:
                    raise ValueError(f"house_prices is missing gazetteer columns: {sorted(missing)}")
                city_state_rows = connection.execute(text(
                    "SELECT DISTINCT city, state FROM public.house_prices ORDER BY city, state"
                )).all()
                addresses = connection.execute(text(
                    "SELECT DISTINCT address FROM public.house_prices WHERE address IS NOT NULL"
                )).scalars().all()
        city_state = {str(city): str(state) for city, state in city_state_rows}
        streets = set()
        for address in addresses:
            normalized = re.sub(r"^\s*\d+\s+", "", str(address)).strip()
            if normalized:
                streets.add(normalized)
        gazetteer = {
            "cities": sorted(city_state),
            "states": sorted(set(city_state.values())),
            "city_state": dict(sorted(city_state.items())),
            "street_names": sorted(streets),
        }
        target = Path(cache_path) if cache_path else _default_cache_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(gazetteer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return gazetteer
    finally:
        engine.dispose()


def load_gazetteer(
    cache_path: str | Path | None = None,
    *,
    refresh: bool = False,
    build_if_missing: bool = True,
) -> dict[str, Any]:
    target = Path(cache_path) if cache_path else _default_cache_path()
    if target.exists() and not refresh:
        payload = json.loads(target.read_text(encoding="utf-8"))
        required_keys = {"cities", "states", "city_state", "street_names"}
        if not required_keys.issubset(payload):
            raise ValueError("Gazetteer cache has an incompatible format.")
        return payload
    if not build_if_missing:
        raise FileNotFoundError(f"Gazetteer cache not found: {target}")
    return build_gazetteer_from_database(target)
