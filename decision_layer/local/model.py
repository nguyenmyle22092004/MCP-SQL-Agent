from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from decision_layer.local.features import FEATURE_NAMES, extract_features

ARTIFACT_VERSION = 1
DEFAULT_L2_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def rule_based_probability(target: str, features: dict[str, float]) -> float:
    """Interpretable binary baseline for either supported model target."""
    if target == "wrong":
        return float(bool(
            features.get("n_unmatched_claims", 0)
            or features.get("n_cities_not_in_evidence", 0)
            or features.get("n_states_not_in_evidence", 0)
            or features.get("n_addresses_not_in_evidence", 0)
            or features.get("city_state_pair_inconsistent", 0)
        ))
    if target == "needs_clarification":
        return float(bool(features.get("n_vague_terms", 0) and not (int(features.get("question_location_flags", 0)) & 0b11)))
    raise ValueError(f"Unsupported rule baseline target: {target}")


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    p = min(1.0 - 1e-8, max(1e-8, probability))
    return math.log(p / (1.0 - p))


def _label(record: dict, target: str) -> int:
    if target == "wrong":
        return int(str(record.get("label", "")).lower() == "wrong")
    return int(bool(record.get("needs_clarification", False)))


def _feature_vector(record: dict, gazetteer: dict[str, Any]) -> dict[str, float]:
    return extract_features(
        str(record.get("question", "")),
        str(record.get("answer", "")),
        record.get("tool_results", []),
        str(record.get("recent_context", "")),
        str(record.get("trusted_context", "")),
        gazetteer,
    )


def _sigmoid_train(
    rows: list[list[float]],
    labels: list[int],
    l2: float,
    *,
    steps: int = 900,
    learning_rate: float = 0.08,
    class_balance: bool = True,
) -> dict:
    if not rows:
        raise ValueError("Cannot train logistic regression without examples.")
    n_features = len(rows[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        prior = (positives + 1.0) / (len(labels) + 2.0)
        return {"intercept": _logit(prior), "coefficients": [0.0] * n_features, "class_weights": {"negative": 1.0, "positive": 1.0}}
    weights = (
        {0: len(labels) / (2.0 * negatives), 1: len(labels) / (2.0 * positives)}
        if class_balance
        else {0: 1.0, 1: 1.0}
    )
    coefficients = [0.0] * n_features
    intercept = _logit((positives + 1.0) / (len(labels) + 2.0))
    for step in range(steps):
        grad = [l2 * weight for weight in coefficients]
        grad_intercept = 0.0
        for row, label in zip(rows, labels):
            probability = _sigmoid(intercept + sum(coef * value for coef, value in zip(coefficients, row)))
            error = (probability - label) * weights[label]
            grad_intercept += error
            for index, value in enumerate(row):
                grad[index] += error * value
        rate = learning_rate / (1.0 + step / 250.0)
        scale = 1.0 / len(rows)
        intercept -= rate * grad_intercept * scale
        coefficients = [coef - rate * gradient * scale for coef, gradient in zip(coefficients, grad)]
    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "class_weights": {"negative": weights[0], "positive": weights[1]},
    }


def _standardize(vectors: list[dict[str, float]], feature_names: tuple[str, ...]) -> tuple[list[dict[str, float]], dict, dict]:
    means = {}
    stds = {}
    for name in feature_names:
        values = [float(vector[name]) for vector in vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        stds[name] = math.sqrt(variance) or 1.0
    standardized = [
        {name: (float(vector[name]) - means[name]) / stds[name] for name in feature_names}
        for vector in vectors
    ]
    return standardized, means, stds


def _as_matrix(vectors: list[dict[str, float]], names: tuple[str, ...]) -> list[list[float]]:
    return [[vector[name] for name in names] for vector in vectors]


def _group_folds(groups: list[str], folds: int, seed: int) -> list[list[int]]:
    unique = list(dict.fromkeys(groups))
    random.Random(seed).shuffle(unique)
    buckets = [[] for _ in range(folds)]
    group_to_fold = {group: index % folds for index, group in enumerate(unique)}
    for index, group in enumerate(groups):
        buckets[group_to_fold[group]].append(index)
    return buckets


def _predict_raw(parameters: dict, vector: list[float]) -> float:
    return _sigmoid(parameters["intercept"] + sum(a * b for a, b in zip(parameters["coefficients"], vector)))


def _log_loss(predictions: list[float], labels: list[int]) -> float:
    eps = 1e-8
    return -sum(y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p)) for p, y in zip(predictions, labels)) / max(len(labels), 1)


def _cross_validated_l2(vectors: list[dict[str, float]], labels: list[int], groups: list[str], names: tuple[str, ...], seed: int, grid: Iterable[float]) -> float:
    n_groups = len(set(groups))
    if n_groups < 2:
        return 0.1
    folds = _group_folds(groups, min(5, n_groups), seed)
    candidates = []
    for l2 in grid:
        all_predictions = [0.5] * len(labels)
        for validation_indices in folds:
            validation_set = set(validation_indices)
            training_indices = [i for i in range(len(labels)) if i not in validation_set]
            if not training_indices or not validation_indices:
                continue
            train_vectors = [vectors[i] for i in training_indices]
            train_labels = [labels[i] for i in training_indices]
            standardized_train, means, stds = _standardize(train_vectors, names)
            model = _sigmoid_train(_as_matrix(standardized_train, names), train_labels, float(l2))
            for index in validation_indices:
                row = [(vectors[index][name] - means[name]) / stds[name] for name in names]
                all_predictions[index] = _predict_raw(model, row)
        candidates.append((_log_loss(all_predictions, labels), float(l2)))
    return min(candidates)[1] if candidates else 0.1


def _fit_platt(scores: list[float], labels: list[int]) -> dict:
    logits = [_logit(score) for score in scores]
    fitted = _sigmoid_train(
        [[value] for value in logits],
        labels,
        l2=0.001,
        steps=1200,
        learning_rate=0.05,
        class_balance=False,
    )
    return {"type": "platt", "slope": fitted["coefficients"][0], "intercept": fitted["intercept"]}


def _fit_isotonic(scores: list[float], labels: list[int]) -> dict:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    blocks = []
    for score, label in pairs:
        blocks.append([score, score, float(label), 1])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    return {"type": "isotonic", "blocks": [[low, high, total / count] for low, high, total, count in blocks]}


def _apply_calibrator(calibrator: dict, probability: float) -> float:
    if calibrator["type"] == "platt":
        return _sigmoid(calibrator["slope"] * _logit(probability) + calibrator["intercept"])
    for low, high, value in calibrator["blocks"]:
        if probability <= high:
            return value
    return calibrator["blocks"][-1][2]


def _fit_target(vectors: list[dict[str, float]], labels: list[int], groups: list[str], names: tuple[str, ...], seed: int, calibration: str, l2_grid: Iterable[float]) -> dict:
    selected_l2 = _cross_validated_l2(vectors, labels, groups, names, seed, l2_grid)
    folds = _group_folds(groups, min(5, len(set(groups))), seed + 1) if len(set(groups)) >= 2 else []
    oof = [0.5] * len(labels)
    for validation_indices in folds:
        validation_set = set(validation_indices)
        training_indices = [i for i in range(len(labels)) if i not in validation_set]
        if not training_indices:
            continue
        train_vectors, train_labels = [vectors[i] for i in training_indices], [labels[i] for i in training_indices]
        standardized_train, means, stds = _standardize(train_vectors, names)
        parameters = _sigmoid_train(_as_matrix(standardized_train, names), train_labels, selected_l2)
        for index in validation_indices:
            row = [(vectors[index][name] - means[name]) / stds[name] for name in names]
            oof[index] = _predict_raw(parameters, row)
    if not folds:
        # Not enough independent groups to form OOF folds; use a smoothed prior calibrator.
        positives = sum(labels)
        prior = (positives + 1.0) / (len(labels) + 2.0)
        calibrator = {"type": "platt", "slope": 0.0, "intercept": _logit(prior)}
    elif calibration == "isotonic":
        calibrator = _fit_isotonic(oof, labels)
    else:
        calibrator = _fit_platt(oof, labels)
    standardized, means, stds = _standardize(vectors, names)
    parameters = _sigmoid_train(_as_matrix(standardized, names), labels, selected_l2)
    calibrated_oof = [_apply_calibrator(calibrator, probability) for probability in oof]
    return {
        "l2": selected_l2,
        "means": means,
        "stds": stds,
        "intercept": parameters["intercept"],
        "coefficients": parameters["coefficients"],
        "class_weights": parameters["class_weights"],
        "calibrator": calibrator,
        "oof_predictions": calibrated_oof,
        "oof_labels": labels,
    }


@dataclass
class LocalModel:
    artifact: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "LocalModel":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if artifact.get("version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported local decision artifact version.")
        if tuple(artifact.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Artifact feature_names do not match current feature code.")
        return cls(artifact)

    def probability(self, target: str, features: dict[str, float]) -> float:
        if tuple(self.artifact.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Artifact feature_names do not match current feature code.")
        model = self.artifact["targets"][target]
        vector = [(float(features[name]) - model["means"][name]) / model["stds"][name] for name in FEATURE_NAMES]
        raw = _predict_raw(model, vector)
        return _apply_calibrator(model["calibrator"], raw)


def train_model(
    examples: list[dict],
    gazetteer: dict[str, Any],
    *,
    seed: int = 17,
    calibration: str = "platt",
    l2_grid: Iterable[float] = DEFAULT_L2_GRID,
) -> dict[str, Any]:
    """Train wrong/clarification models on train-split examples only."""
    if calibration not in {"platt", "isotonic"}:
        raise ValueError("calibration must be 'platt' or 'isotonic'.")
    train_records = [record for record in examples if record.get("split") == "train"]
    if not train_records:
        raise ValueError("No train-split labeled examples were provided.")
    vectors = [_feature_vector(record, gazetteer) for record in train_records]
    if any(tuple(vector) != FEATURE_NAMES for vector in vectors):
        raise ValueError("Feature extractor returned an incompatible feature schema.")
    groups = []
    for index, record in enumerate(train_records):
        group = record.get("group_id") or record.get("template_id") or record.get("source_row_id")
        if not group:
            raise ValueError(f"Train row {index} is missing group_id/template_id/source_row_id for leakage-safe folds.")
        groups.append(str(group))
    targets = {}
    for offset, (target_name, target_key) in enumerate((("wrong", "wrong"), ("needs_clarification", "needs_clarification"))):
        labels = [_label(record, target_key) for record in train_records]
        targets[target_name] = _fit_target(vectors, labels, groups, FEATURE_NAMES, seed + offset, calibration, l2_grid)
    canonical_training = json.dumps(train_records, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return {
        "version": ARTIFACT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "targets": targets,
        "calibration": calibration,
        "training_data_sha256": hashlib.sha256(canonical_training).hexdigest(),
        "training_rows": len(train_records),
        "train_groups": groups,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
