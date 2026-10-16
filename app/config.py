from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg://agent:agent@localhost:5433/house_price_agent"
    READONLY_DATABASE_URL: str = ""
    READONLY_DB_USER: str = ""
    READONLY_DB_PASSWORD: str = ""
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-5"
    PROMPT_VERSION: str = "sql-planner-v2"
    DECISION_BACKEND: str = "fake"
    DECISION_LAYER_ENABLED: bool = False
    DECISION_FAIL_CLOSED: bool = False
    DECISION_MODEL_PATH: str = "decision_layer/artifacts/local_model.json"
    DECISION_GAZETTEER_PATH: str = "data/gazetteer.json"
    DECISION_FALSE_ACCEPT_COST: float = 10.0
    DECISION_FALSE_REJECT_COST: float = 1.0
    DECISION_CLARIFICATION_COST: float = 2.0
    DECISION_WRONG_THRESHOLD: float = 0.5
    DECISION_ENTITY_MATCH_PROBABILITY: float = 0.02
    DECISION_ENTITY_MISMATCH_PROBABILITY: float = 0.98
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_PROMPT_CACHE: bool = True
    AGENT_MAX_ITERATIONS: int = 3
    AGENT_TIMEOUT_SECONDS: float = 45.0
    AGENT_MAX_TOKENS: int = 12000
    AGENT_MAX_COST_USD: float = 0.05
    LLM_INPUT_COST_PER_MILLION_USD: float = 3.0
    LLM_OUTPUT_COST_PER_MILLION_USD: float = 15.0
    SQL_TOOL_TIMEOUT_SECONDS: float = 10.0
    MCP_TOOL_TIMEOUT_SECONDS: float = 12.0
    APP_ENV: str = "development"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_readonly_database_url(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.READONLY_DATABASE_URL.strip():
        return settings.READONLY_DATABASE_URL
    if settings.READONLY_DB_USER and settings.READONLY_DB_PASSWORD:
        return make_url(settings.DATABASE_URL).set(
            username=settings.READONLY_DB_USER,
            password=settings.READONLY_DB_PASSWORD,
        ).render_as_string(hide_password=False)
    if settings.APP_ENV.strip().lower() in {"prod", "production"}:
        raise ValueError("Production requires dedicated read-only database credentials.")
    return settings.DATABASE_URL
