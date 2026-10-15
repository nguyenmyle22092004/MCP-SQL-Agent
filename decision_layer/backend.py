from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from decision_layer.questions import Choice, Predicate, Score

DecisionQuestion: TypeAlias = Predicate | Choice | Score


@dataclass(frozen=True)
class DecisionAnswer:
    value: bool | str | float | dict[str, float] | None = None
    error: str | None = None
    refusal: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None or self.refusal is not None


class DecisionBackend(Protocol):
    def answer(self, question: DecisionQuestion) -> DecisionAnswer:
        """Return a structured decision or a fail-closed error/refusal."""


class FakeBackend:
    """Deterministic non-learning backend used as the compatibility default."""

    def answer(self, question: DecisionQuestion) -> DecisionAnswer:
        features = question.features
        if isinstance(question, Predicate):
            if question.name == "entity_grounding":
                return DecisionAnswer(value=bool(
                    features.get("n_cities_not_in_evidence", 0)
                    or features.get("n_states_not_in_evidence", 0)
                    or features.get("n_addresses_not_in_evidence", 0)
                    or features.get("city_state_pair_inconsistent", 0)
                ))
            if question.name == "needs_clarification":
                vague_without_city = features.get("n_vague_terms", 0) > 0 and not features.get("question_mentions_city", 0)
                return DecisionAnswer(value=vague_without_city)
            if question.name == "answer_wrong":
                return DecisionAnswer(value=bool(
                    features.get("n_unmatched_claims", 0)
                    or features.get("n_cities_not_in_evidence", 0)
                    or features.get("n_states_not_in_evidence", 0)
                    or features.get("n_addresses_not_in_evidence", 0)
                    or features.get("city_state_pair_inconsistent", 0)
                ))
            return DecisionAnswer(error=f"Unsupported predicate: {question.name}")
        if isinstance(question, Choice):
            chosen = features.get("question_type_code", 5.0)
            type_names = ("count", "average", "ranking", "lookup", "comparison", "other")
            index = max(0, min(int(chosen), len(type_names) - 1))
            label = type_names[index]
            probabilities = {choice: float(choice == label) for choice in question.choices}
            if not any(probabilities.values()) and "other" in question.choices:
                probabilities = {choice: float(choice == "other") for choice in question.choices}
            return DecisionAnswer(value=label if label in question.choices else "other", probabilities=probabilities)
        if isinstance(question, Score):
            return DecisionAnswer(error="FakeBackend does not implement Score questions.")
        return DecisionAnswer(error="Unsupported decision question type.")
