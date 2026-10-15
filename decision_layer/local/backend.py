from __future__ import annotations

from pathlib import Path
from typing import Any

from decision_layer.backend import DecisionAnswer
from decision_layer.local.features import FEATURE_NAMES
from decision_layer.local.model import LocalModel
from decision_layer.questions import Choice, Predicate, Score


class LocalBackend:
    def __init__(
        self,
        artifact_path: str | Path = "decision_layer/artifacts/local_model.json",
        *,
        entity_match_probability: float = 0.02,
        entity_mismatch_probability: float = 0.98,
    ):
        self.artifact_path = Path(artifact_path)
        self.entity_match_probability = float(entity_match_probability)
        self.entity_mismatch_probability = float(entity_mismatch_probability)
        if not 0.0 <= self.entity_match_probability <= 1.0 or not 0.0 <= self.entity_mismatch_probability <= 1.0:
            raise ValueError("Entity probabilities must be between 0 and 1.")
        self._model: LocalModel | None = None
        self._load_error: str | None = None
        try:
            self._model = LocalModel.from_file(self.artifact_path)
        except Exception as exc:
            self._load_error = type(exc).__name__

    def answer(self, question: Predicate | Choice | Score) -> DecisionAnswer:
        if isinstance(question, Score):
            raise NotImplementedError("LocalBackend does not implement Score questions; use the p_wrong predicate fallback.")
        features = question.features
        if isinstance(question, Predicate) and question.name == "entity_grounding":
            mismatch = bool(
                features.get("n_cities_not_in_evidence", 0)
                or features.get("n_states_not_in_evidence", 0)
                or features.get("n_addresses_not_in_evidence", 0)
                or features.get("city_state_pair_inconsistent", 0)
            )
            probability = self.entity_mismatch_probability if mismatch else self.entity_match_probability
            return DecisionAnswer(value=probability)
        if isinstance(question, Choice) and question.name == "question_type":
            names = ("count", "average", "ranking", "lookup", "comparison", "other")
            index = max(0, min(int(features.get("question_type_code", 5.0)), len(names) - 1))
            label = names[index]
            probabilities = {
                choice: (0.98 if choice == label else 0.02 / max(len(question.choices) - 1, 1))
                for choice in question.choices
            }
            if label not in question.choices:
                label = "other"
                probabilities = {choice: 0.0 for choice in question.choices}
                if "other" in probabilities:
                    probabilities["other"] = 1.0
                else:
                    probabilities["other"] = 1.0
            return DecisionAnswer(value=label, probabilities=probabilities)
        if self._model is None:
            return DecisionAnswer(error=f"Local model unavailable ({self._load_error or 'load failed'}).")
        if isinstance(question, Predicate):
            if question.name == "needs_clarification":
                return DecisionAnswer(value=self._model.probability("needs_clarification", features))
            if question.name == "answer_wrong":
                return DecisionAnswer(value=self._model.probability("wrong", features))
            return DecisionAnswer(error=f"Unsupported predicate: {question.name}")
        return DecisionAnswer(error="Unsupported decision question type.")
