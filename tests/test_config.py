import pytest
from sqlalchemy.engine import make_url

from app.config import Settings, get_readonly_database_url


def test_production_requires_dedicated_readonly_database_credentials():
    settings = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql+psycopg://agent:write-password@localhost/house_price_agent",
    )

    with pytest.raises(ValueError, match="read-only database credentials"):
        get_readonly_database_url(settings)


def test_production_accepts_explicit_readonly_database_url():
    readonly_url = "postgresql+psycopg://reader:read-password@localhost/house_price_agent"
    settings = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql+psycopg://agent:write-password@localhost/house_price_agent",
        READONLY_DATABASE_URL=readonly_url,
    )

    assert get_readonly_database_url(settings) == readonly_url


def test_production_can_derive_url_from_readonly_user_credentials():
    settings = Settings(
        APP_ENV="prod",
        DATABASE_URL="postgresql+psycopg://agent:write-password@localhost/house_price_agent",
        READONLY_DB_USER="reader",
        READONLY_DB_PASSWORD="read-password",
    )

    readonly_url = get_readonly_database_url(settings)

    parsed_url = make_url(readonly_url)
    assert parsed_url.username == "reader"
    assert parsed_url.password == "read-password"


def test_development_can_fall_back_to_application_database_url():
    database_url = "postgresql+psycopg://agent:write-password@localhost/house_price_agent"
    settings = Settings(APP_ENV="development", DATABASE_URL=database_url)

    assert get_readonly_database_url(settings) == database_url