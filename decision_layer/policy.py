from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from time import perf_counter
from typing import Any

from app.tracing.validator import validate_answer
from decision_layer.backend import DecisionAnswer, DecisionBackend
from decision_layer.questions import Choice, Predicate, Score


class Action(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    CLARIFY = "clarify"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CostConfig:
    false_accept: float = 10.0
    false_reject: float = 1.0
    clarification: float = 2.0
    wrong_threshold: float = 0.5

    def __post_init__(self) -> None:
        if min(self.false_accept, self.false_reject, self.clarification) < 0:
            raise ValueError("Decision costs must be non-negative.")
        if not 0.0 <= self.wrong_threshold <= 1.0:
            raise ValueError("wrong_threshold must be between 0 and 1.")


def _answer_probability(answer: DecisionAnswer) -> float | None:
    if answer.failed:
        return None
    if isinstance(answer.value, bool):
        return 1.0 if answer.value else 0.0
    if isinstance(answer.value, (int, float)):
        return max(0.0, min(float(answer.value), 1.0))
    return None


def choose_action(p_wrong: float, p_clarification: float, costs: CostConfig) -> Action:
    p_wrong = max(0.0, min(float(p_wrong), 1.0))
    p_clarification = max(0.0, min(float(p_clarification), 1.0))
    cost_threshold = costs.false_reject / max(costs.false_accept + costs.false_reject, 1e-12)
    effective_threshold = min(max(0.0, min(costs.wrong_threshold, 1.0)), cost_threshold)
    accept_cost = p_wrong * costs.false_accept
    reject_cost = (1.0 - p_wrong) * costs.false_reject
    if p_wrong > effective_threshold:
        reject_cost = min(reject_cost, accept_cost - 1e-12)
    clarify_cost = costs.clarification + min(accept_cost, reject_cost) * (1.0 - p_clarification)
    if clarify_cost < min(accept_cost, reject_cost):
        return Action.CLARIFY
    if accept_cost <= reject_cost:
        return Action.ACCEPT
    return Action.REJECT


def _safe_feature_summary(features: dict[str, float]) -> dict[str, float]:
    return {name: round(float(value), 6) for name, value in features.items()}


def _answer_wrong_probability(backend: DecisionBackend, features: dict[str, float]) -> float | None:
    try:
        scored = backend.answer(Score("answer_wrong", features))
    except NotImplementedError:
        scored = DecisionAnswer(error="Score is unsupported")
    probability = _answer_probability(scored)
    if probability is not None:
        return probability
    return _answer_probability(backend.answer(Predicate("answer_wrong", features)))


def validate_with_decisions(
    question: str,
    answer: str,
    tool_results: list[dict] | None,
    recent_context: str,
    trusted_context: str,
    gazetteer: dict[str, Any],
    backend: DecisionBackend,
    costs: CostConfig | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run the immutable deterministic validator first, then optional decision policy."""
    deterministic = validate_answer(question, answer, tool_results, recent_context, trusted_context)
    if deterministic["status"] == "failed":
        deterministic.update({
            "action": Action.REJECT.value,
            "decision_status": "deterministic_reject",
            "latency_ms": 0.0,
            "features": {},
            "probabilities": {},
        })
        if logger is not None:
            safe_payload = {
                "action": Action.REJECT.value,
                "decision_status": "deterministic_reject",
                "deterministic_status": deterministic["status"],
                "n_errors": len(deterministic["errors"]),
                "features": {},
                "probabilities": {},
                "latency_ms": 0.0,
                "evidence_sha256": evidence_hash(tool_results),
            }
            if hasattr(logger, "add_event"):
                logger.add_event("DECISION_LAYER", **safe_payload)
            elif callable(logger):
                logger(safe_payload)
        return deterministic

    started = perf_counter()
    features: dict[str, float] = {}
    try:
        from decision_layer.local.features import extract_features

        features = extract_features(question, answer, tool_results or [], recent_context, trusted_context, gazetteer)
        p_wrong = _answer_wrong_probability(backend, features)
        clarification = backend.answer(Predicate("needs_clarification", features))
        entity_grounding = backend.answer(Predicate("entity_grounding", features))
        p_clarification = _answer_probability(clarification)
        p_entity_mismatch = _answer_probability(entity_grounding)
        if p_wrong is None or p_clarification is None or p_entity_mismatch is None:
            raise RuntimeError("Decision backend returned an error/refusal or unsupported answer.")
        type_answer = backend.answer(Choice("question_type", ("count", "average", "ranking", "lookup", "comparison", "other"), features))
        if type_answer.failed:
            raise RuntimeError("Decision backend refused the question_type choice.")
        policy_costs = costs or CostConfig()
        action = choose_action(p_wrong, p_clarification, policy_costs)
        result = {
            **deterministic,
            "action": action.value,
            "decision_status": "ok",
            "probabilities": {
                "answer_wrong": p_wrong,
                "needs_clarification": p_clarification,
                "entity_mismatch": p_entity_mismatch,
            },
            "question_type": type_answer.value,
            "features": _safe_feature_summary(features),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:
        result = {
            **deterministic,
            "action": Action.SKIPPED.value,
            "decision_status": "skipped",
            "decision_error": type(exc).__name__,
            "features": _safe_feature_summary(features),
            "probabilities": {},
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        }

    if logger is not None:
        if hasattr(logger, "add_event"):
            logger.add_event(
                "DECISION_LAYER",
                **{k: v for k, v in result.items() if k not in {"expected", "actual", "claims", "errors"}},
                evidence_sha256=evidence_hash(tool_results),
            )
        elif callable(logger):
            safe_payload = {k: v for k, v in result.items() if k not in {"expected", "actual", "claims", "errors"}}
            safe_payload["evidence_sha256"] = evidence_hash(tool_results)
            logger(safe_payload)
    return result


def evidence_hash(tool_results: list[dict] | None) -> str:
    safe = []
    for item in tool_results or []:
        evidence = item.get("evidence", {}) if isinstance(item, dict) else {}
        safe.append({"tool": item.get("tool"), "sql": item.get("sql") or evidence.get("sql"), "evidence_id": evidence.get("evidence_id")})
    return sha256(json.dumps(safe, sort_keys=True, default=str).encode("utf-8")).hexdigest()
