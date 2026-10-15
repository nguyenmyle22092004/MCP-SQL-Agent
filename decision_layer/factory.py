from __future__ import annotations

from app.config import Settings, get_settings
from decision_layer.backend import DecisionAnswer, DecisionBackend, FakeBackend
from decision_layer.local.backend import LocalBackend


class UnavailableBackend:
    def __init__(self, name: str):
        self.name = name

    def answer(self, question) -> DecisionAnswer:
        return DecisionAnswer(error=f"Decision backend '{self.name}' has no adapter configured.")


def build_backend(settings: Settings | None = None) -> DecisionBackend:
    settings = settings or get_settings()
    name = settings.DECISION_BACKEND.strip().lower()
    if name == "fake":
        return FakeBackend()
    if name == "local":
        return LocalBackend(
            settings.DECISION_MODEL_PATH,
            entity_match_probability=settings.DECISION_ENTITY_MATCH_PROBABILITY,
            entity_mismatch_probability=settings.DECISION_ENTITY_MISMATCH_PROBABILITY,
        )
    if name in {"openai", "jev"}:
        return UnavailableBackend(name)
    return UnavailableBackend(name or "unset")
