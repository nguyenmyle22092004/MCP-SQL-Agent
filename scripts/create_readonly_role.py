from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    role_name = settings.READONLY_DB_USER
    role_password = settings.READONLY_DB_PASSWORD
    if not role_name or not role_password:
        raise SystemExit("Set READONLY_DB_USER and READONLY_DB_PASSWORD in .env first.")
    if role_name == make_url(settings.DATABASE_URL).username:
        raise SystemExit("The read-only role must be different from the application/database owner role.")

    admin_url = make_url(settings.DATABASE_URL).set(drivername="postgresql")
    database_name = admin_url.database
    with psycopg.connect(str(admin_url), autocommit=True) as connection:
        exists = connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role_name,)).fetchone()
        if exists:
            connection.execute(
                sql.SQL(
                    "ALTER ROLE {} WITH LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT"
                ).format(sql.Identifier(role_name), sql.Literal(role_password))
            )
        else:
            connection.execute(
                sql.SQL(
                    "CREATE ROLE {} WITH LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT"
                ).format(sql.Identifier(role_name), sql.Literal(role_password))
            )
        connection.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(database_name), sql.Identifier(role_name)
            )
        )
        connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role_name))
        )
        connection.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {}").format(
                sql.Identifier(role_name)
            )
        )
        connection.execute(
            sql.SQL("GRANT SELECT ON TABLE public.house_prices TO {}").format(sql.Identifier(role_name))
        )
        connection.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM {}").format(
                sql.Identifier(role_name)
            )
        )

    print(f"Read-only database role '{role_name}' is ready for public.house_prices.")
    print("SQL MCP will use this role automatically when READONLY_DB_USER and READONLY_DB_PASSWORD are set.")


if __name__ == "__main__":
    main()
