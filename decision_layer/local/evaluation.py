from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any

from app.tracing.validator import validate_answer
from decision_layer.local.features import FEATURE_NAMES, extract_features
from decision_layer.local.model import LocalModel, rule_based_probability, train_model


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at line {line_number}.")
                records.append(record)
    return records


def _binary_label(record: dict) -> int:
    return int(str(record.get("label", "")).lower() == "wrong")


def _group_key(record: dict, index: int) -> str:
    group = record.get("group_id") or record.get("template_id") or record.get("source_row_id")
    if not group:
        raise ValueError(f"Labeled row {index} is missing group_id/template_id/source_row_id for leakage-safe folds.")
    return str(group)


def _confusion(labels: list[int], predictions: list[int]) -> dict[str, int]:
    return {
        "true_positive": sum(y == 1 and p == 1 for y, p in zip(labels, predictions)),
        "false_positive": sum(y == 0 and p == 1 for y, p in zip(labels, predictions)),
        "true_negative": sum(y == 0 and p == 0 for y, p in zip(labels, predictions)),
        "false_negative": sum(y == 1 and p == 0 for y, p in zip(labels, predictions)),
    }


def _auc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(probabilities, labels))
    rank_sum = 0.0
    position = 1
    while position <= len(pairs):
        end = position
        while end < len(pairs) and pairs[end][0] == pairs[position - 1][0]:
            end += 1
        average_rank = (position + end) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[position - 1:end])
        position = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _brier(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels:
        return None
    return sum((p - label) ** 2 for label, p in zip(labels, probabilities)) / len(labels)


def _reliability(labels: list[int], probabilities: list[float], bins: int = 10) -> list[dict[str, float | int]]:
    table = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        indices = [i for i, p in enumerate(probabilities) if low <= p < high or (index == bins - 1 and p == 1.0)]
        if indices:
            table.append({
                "bin": index,
                "mean_predicted": sum(probabilities[i] for i in indices) / len(indices),
                "observed_rate": sum(labels[i] for i in indices) / len(indices),
                "count": len(indices),
            })
    return table


def _ece(labels: list[int], probabilities: list[float], bins: int = 10) -> float | None:
    if not labels:
        return None
    table = _reliability(labels, probabilities, bins)
    return sum(row["count"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in table) / len(labels)


def _bootstrap_ci(
    labels: list[int],
    probabilities: list[float],
    *,
    metric: str,
    seed: int,
    threshold: float = 0.5,
    deterministic_rejects: list[int] | None = None,
    resamples: int = 1000,
) -> tuple[float, float] | None:
    if not labels:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        indices = [rng.randrange(len(labels)) for _ in labels]
        sample_labels = [labels[i] for i in indices]
        sample_probs = [probabilities[i] for i in indices]
        sample_rejects = [deterministic_rejects[i] for i in indices] if deterministic_rejects is not None else [0] * len(indices)
        if metric == "false_accept":
            wrong = sum(sample_labels)
            if not wrong:
                continue
            value = sum(y == 1 and not deterministic and p < threshold for y, deterministic, p in zip(sample_labels, sample_rejects, sample_probs)) / wrong
        elif metric == "auc":
            value = _auc(sample_labels, sample_probs)
            if value is None:
                continue
        elif metric == "brier":
            value = _brier(sample_labels, sample_probs)
            if value is None:
                continue
        else:
            value = _ece(sample_labels, sample_probs)
            if value is None:
                continue
        values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _predict_record(model: LocalModel, record: dict, gazetteer: dict) -> tuple[dict, dict[str, float]]:
    features = extract_features(
        str(record.get("question", "")),
        str(record.get("answer", "")),
        record.get("tool_results", []),
        str(record.get("recent_context", "")),
        str(record.get("trusted_context", "")),
        gazetteer,
    )
    return features, {target: model.probability(target, features) for target in ("wrong", "needs_clarification")}


def _choose_train_threshold(
    artifact: dict,
    false_accept_cost: float,
    false_reject_cost: float,
    deterministic_rejects: list[int],
) -> tuple[float, float, list[dict[str, float]]]:
    target = artifact["targets"]["wrong"]
    probabilities = target.get("oof_predictions", [])
    labels = target.get("oof_labels", [])
    candidates = [i / 100 for i in range(101)]
    scored = []
    sweep = []
    for threshold in candidates:
        false_accept_count = sum(y == 1 and not rejected and p < threshold for y, rejected, p in zip(labels, deterministic_rejects, probabilities))
        false_reject_count = sum(y == 0 and (rejected or p >= threshold) for y, rejected, p in zip(labels, deterministic_rejects, probabilities))
        false_accept = false_accept_count / max(sum(labels), 1)
        false_reject = false_reject_count / max(len(labels) - sum(labels), 1)
        cost = (false_accept_cost * false_accept_count + false_reject_cost * false_reject_count) / max(len(labels), 1)
        scored.append((cost, threshold))
        sweep.append({"threshold": threshold, "false_accept": false_accept, "false_reject": false_reject, "expected_cost": cost})
    cost, threshold = min(scored)
    return threshold, cost, sweep


def evaluate_records(
    records: list[dict],
    gazetteer: dict,
    *,
    false_accept_cost: float = 10.0,
    false_reject_cost: float = 1.0,
    seed: int = 17,
    bootstrap_resamples: int = 1000,
) -> tuple[dict, str]:
    train = [record for record in records if record.get("split") == "train"]
    test = [record for record in records if record.get("split") == "test"]
    if not train or not test:
        raise ValueError("Labeled dataset must contain both train and test split rows.")
    train_groups = {_group_key(record, index) for index, record in enumerate(train)}
    test_groups = {_group_key(record, index) for index, record in enumerate(test)}
    if train_groups.intersection(test_groups):
        raise ValueError("A group_id/template_id/source_row_id occurs in both train and test splits.")
    artifact = train_model(train, gazetteer, seed=seed)
    model = LocalModel(artifact)
    train_deterministic_rejects = [
        int(validate_answer(
            str(row.get("question", "")),
            str(row.get("answer", "")),
            row.get("tool_results", []),
            str(row.get("recent_context", "")),
            str(row.get("trusted_context", "")),
        )["status"] == "failed")
        for row in train
    ]
    threshold, train_cost, threshold_sweep = _choose_train_threshold(
        artifact, false_accept_cost, false_reject_cost, train_deterministic_rejects
    )
    labels = [_binary_label(record) for record in test]
    features_and_predictions = [_predict_record(model, record, gazetteer) for record in test]
    features = [item[0] for item in features_and_predictions]
    wrong_probabilities = [item[1]["wrong"] for item in features_and_predictions]
    model_predictions = [int(p >= threshold) for p in wrong_probabilities]
    deterministic_predictions = [
        int(validate_answer(
            str(row.get("question", "")),
            str(row.get("answer", "")),
            row.get("tool_results", []),
            str(row.get("recent_context", "")),
            str(row.get("trusted_context", "")),
        )["status"] == "failed")
        for row in test
    ]
    rule_predictions = [int(rule_based_probability("wrong", feature) >= 0.5) for feature in features]
    combined_predictions = [
        int(deterministic or model_prediction)
        for deterministic, model_prediction in zip(deterministic_predictions, model_predictions)
    ]

    report: dict[str, Any] = {
        "test_rows": len(test),
        "train_rows": len(train),
        "wrong_case_count_by_split": {
            "train": sum(_binary_label(record) for record in train),
            "test": sum(_binary_label(record) for record in test),
        },
        "threshold_chosen_on_train_oof": threshold,
        "train_oof_expected_cost": train_cost,
        "train_threshold_sweep": threshold_sweep,
        "test": {},
    }
    for name, predictions in (
        ("deterministic", deterministic_predictions),
        ("rules", rule_predictions),
        ("model", model_predictions),
        ("combined", combined_predictions),
    ):
        report["test"][name] = {"confusion": _confusion(labels, predictions)}
    report["test"]["model"].update({
        "auc": _auc(labels, wrong_probabilities),
        "brier": _brier(labels, wrong_probabilities),
        "ece": _ece(labels, wrong_probabilities),
        "reliability": _reliability(labels, wrong_probabilities),
        "bootstrap_false_accept_ci_95": _bootstrap_ci(
            labels,
            wrong_probabilities,
            metric="false_accept",
            seed=seed,
            threshold=threshold,
            deterministic_rejects=deterministic_predictions,
            resamples=bootstrap_resamples,
        ),
        "bootstrap_ece_ci_95": _bootstrap_ci(labels, wrong_probabilities, metric="ece", seed=seed + 1, resamples=bootstrap_resamples),
        "bootstrap_auc_ci_95": _bootstrap_ci(labels, wrong_probabilities, metric="auc", seed=seed + 2, resamples=bootstrap_resamples),
        "bootstrap_brier_ci_95": _bootstrap_ci(labels, wrong_probabilities, metric="brier", seed=seed + 3, resamples=bootstrap_resamples),
    })
    report["test"]["combined"]["expected_cost"] = (
        false_accept_cost * sum(y == 1 and not pred for y, pred in zip(labels, combined_predictions)) / len(test)
        + false_reject_cost * sum(y == 0 and pred for y, pred in zip(labels, combined_predictions)) / len(test)
    )

    by_source = {}
    for source in sorted({str(row.get("source", "unknown")) for row in test}):
        indexes = [index for index, row in enumerate(test) if str(row.get("source", "unknown")) == source]
        source_labels = [labels[i] for i in indexes]
        source_probs = [wrong_probabilities[i] for i in indexes]
        by_source[source] = {
            "count": len(indexes),
            "deterministic_confusion": _confusion(source_labels, [deterministic_predictions[i] for i in indexes]),
            "rule_confusion": _confusion(source_labels, [rule_predictions[i] for i in indexes]),
            "model_confusion": _confusion(source_labels, [model_predictions[i] for i in indexes]),
            "combined_confusion": _confusion(source_labels, [combined_predictions[i] for i in indexes]),
            "auc": _auc(source_labels, source_probs),
            "brier": _brier(source_labels, source_probs),
            "ece": _ece(source_labels, source_probs),
        }
    report["by_source"] = by_source
    error_recall = {}
    for error_type in sorted({str(row.get("error_type", "unknown")) for row, label in zip(test, labels) if label}):
        indexes = [i for i, (row, label) in enumerate(zip(test, labels)) if label and str(row.get("error_type", "unknown")) == error_type]
        error_recall[error_type] = sum(model_predictions[i] for i in indexes) / len(indexes)
    report["wrong_recall_by_error_type"] = error_recall
    clarification_labels = [int(bool(row.get("needs_clarification", False))) for row in test]
    clarification_probabilities = [item[1]["needs_clarification"] for item in features_and_predictions]
    clarification_rule_predictions = [int(rule_based_probability("needs_clarification", feature) >= 0.5) for feature in features]
    report["needs_clarification"] = {
        "rule_baseline_confusion": _confusion(clarification_labels, clarification_rule_predictions),
        "model_auc": _auc(clarification_labels, clarification_probabilities),
        "model_brier": _brier(clarification_labels, clarification_probabilities),
        "model_ece": _ece(clarification_labels, clarification_probabilities),
    }
    report["needs_clarification"]["by_source"] = {}
    for source in sorted({str(row.get("source", "unknown")) for row in test}):
        indexes = [index for index, row in enumerate(test) if str(row.get("source", "unknown")) == source]
        source_labels = [clarification_labels[i] for i in indexes]
        source_probabilities = [clarification_probabilities[i] for i in indexes]
        report["needs_clarification"]["by_source"][source] = {
            "count": len(indexes),
            "rule_confusion": _confusion(source_labels, [clarification_rule_predictions[i] for i in indexes]),
            "model_confusion": _confusion(source_labels, [int(clarification_probabilities[i] >= 0.5) for i in indexes]),
            "auc": _auc(source_labels, source_probabilities),
            "brier": _brier(source_labels, source_probabilities),
            "ece": _ece(source_labels, source_probabilities),
        }
    report["wrong_case_count"] = sum(labels)
    report["warning"] = "Fewer than 30 wrong test cases; uncertainty is high." if sum(labels) < 30 else None
    report["feature_importance"] = {
        target: sorted(
            zip(FEATURE_NAMES, artifact["targets"][target]["coefficients"]),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        for target in ("wrong", "needs_clarification")
    }

    shuffled = [dict(row) for row in train]
    shuffled_labels = [row.get("label") for row in shuffled]
    random.Random(seed + 2).shuffle(shuffled_labels)
    for row, label in zip(shuffled, shuffled_labels):
        row["label"] = label
    shuffled_artifact = train_model(shuffled, gazetteer, seed=seed + 2)
    shuffled_model = LocalModel(shuffled_artifact)
    shuffled_probabilities = [_predict_record(shuffled_model, row, gazetteer)[1]["wrong"] for row in test]
    report["sanity_shuffled_label_auc"] = _auc(labels, shuffled_probabilities)
    shuffled_auc = report["sanity_shuffled_label_auc"]
    report["sanity_shuffled_label_near_random"] = shuffled_auc is not None and abs(shuffled_auc - 0.5) <= 0.25

    real_test = [row for row in test if row.get("source") == "real"]
    synthetic_train = [row for row in train if row.get("source") == "synthetic"]
    if real_test and synthetic_train and len({row.get("group_id", row.get("template_id", id(row))) for row in synthetic_train}) >= 2:
        synthetic_artifact = train_model(synthetic_train, gazetteer, seed=seed + 3)
        synthetic_model = LocalModel(synthetic_artifact)
        real_probs = [_predict_record(synthetic_model, row, gazetteer)[1]["wrong"] for row in real_test]
        real_labels = [_binary_label(row) for row in real_test]
        all_train_real_probs = [_predict_record(model, row, gazetteer)[1]["wrong"] for row in real_test]
        synthetic_auc = _auc(real_labels, real_probs)
        all_train_auc = _auc(real_labels, all_train_real_probs)
        synthetic_ece = _ece(real_labels, real_probs)
        all_train_ece = _ece(real_labels, all_train_real_probs)
        report["synthetic_train_real_test"] = {
            "count": len(real_test),
            "auc": synthetic_auc,
            "ece": synthetic_ece,
            "auc_gap_vs_all_train": None if synthetic_auc is None or all_train_auc is None else synthetic_auc - all_train_auc,
            "ece_gap_vs_all_train": None if synthetic_ece is None or all_train_ece is None else synthetic_ece - all_train_ece,
        }
    else:
        report["synthetic_train_real_test"] = None

    deterministic_false_accept = report["test"]["deterministic"]["confusion"]["false_negative"]
    combined_false_accept = report["test"]["combined"]["confusion"]["false_negative"]
    deterministic_false_reject = report["test"]["deterministic"]["confusion"]["false_positive"]
    combined_false_reject = report["test"]["combined"]["confusion"]["false_positive"]
    beats = combined_false_accept < deterministic_false_accept and combined_false_reject <= deterministic_false_reject + max(1, round(0.1 * len(test)))
    report["verdict"] = "Model + deterministic validator reduces false accepts without a large false-reject increase." if beats else "Model does not yet demonstrate a reliable improvement over deterministic validation."
    markdown = render_markdown(report)
    return report, markdown


def render_markdown(report: dict) -> str:
    lines = [
        "# Local Decision Model Evaluation",
        "",
        f"Train rows: {report['train_rows']} | Test rows: {report['test_rows']}",
        f"Train-OOF cost-optimal threshold: {report['threshold_chosen_on_train_oof']:.2f} (train cost {report['train_oof_expected_cost']:.4f})",
        "",
        "## Test Metrics",
        "",
        "| System | TP | FP | TN | FN | AUC | Brier | ECE | Expected cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["test"].items():
        confusion = row["confusion"]
        auc = "n/a" if row.get("auc") is None else f"{row['auc']:.3f}"
        brier = "n/a" if row.get("brier") is None else f"{row['brier']:.3f}"
        ece = "n/a" if row.get("ece") is None else f"{row['ece']:.3f}"
        cost = "n/a" if row.get("expected_cost") is None else f"{row['expected_cost']:.3f}"
        lines.append(f"| {name} | {confusion['true_positive']} | {confusion['false_positive']} | {confusion['true_negative']} | {confusion['false_negative']} | {auc} | {brier} | {ece} | {cost} |")
    lines.extend(["", "## Reliability (Model)", "", "| Bin | Mean predicted | Observed rate | Count |", "|---:|---:|---:|---:|"])
    for row in report["test"]["model"]["reliability"]:
        lines.append(f"| {row['bin']} | {row['mean_predicted']:.3f} | {row['observed_rate']:.3f} | {row['count']} |")
    lines.extend([
        "",
        "## Threshold Sweep (Train OOF)",
        "",
        "| Threshold | False accept | False reject | Expected cost |",
        "|---:|---:|---:|---:|",
    ])
    for row in report["train_threshold_sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['false_accept']:.3f} | {row['false_reject']:.3f} | {row['expected_cost']:.3f} |")
    lines.extend([
        "",
        "## By Source",
        "",
        "Confusion matrices are included for deterministic-only, rule-only, model-only, and combined gated decisions.",
        "",
        "| Source | System | TP | FP | TN | FN | AUC | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for source, row in report["by_source"].items():
        for system in ("deterministic", "rule", "model", "combined"):
            confusion = row[f"{system}_confusion"]
            auc = row["auc"] if system == "model" else None
            brier = row["brier"] if system == "model" else None
            ece = row["ece"] if system == "model" else None
            formatted = ["n/a" if value is None else f"{value:.3f}" for value in (auc, brier, ece)]
            lines.append(f"| {source} | {system} | {confusion['true_positive']} | {confusion['false_positive']} | {confusion['true_negative']} | {confusion['false_negative']} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    lines.extend(["", "## Wrong-Case Recall by Error Type", ""])
    for error_type, recall in report["wrong_recall_by_error_type"].items():
        lines.append(f"- `{error_type}`: {recall:.3f}")
    lines.extend(["", "## Needs-Clarification Target", "", f"Rule baseline confusion: {report['needs_clarification']['rule_baseline_confusion']}", f"Model AUC/Brier/ECE: {report['needs_clarification']['model_auc']} / {report['needs_clarification']['model_brier']} / {report['needs_clarification']['model_ece']}", ""])
    lines.extend([
        "| Source | Count | Rule TP | Rule FP | Rule TN | Rule FN | Model AUC | Model Brier | Model ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for source, row in report["needs_clarification"]["by_source"].items():
        confusion = row["rule_confusion"]
        vals = ["n/a" if row[key] is None else f"{row[key]:.3f}" for key in ("auc", "brier", "ece")]
        lines.append(f"| {source} | {row['count']} | {confusion['true_positive']} | {confusion['false_positive']} | {confusion['true_negative']} | {confusion['false_negative']} | {vals[0]} | {vals[1]} | {vals[2]} |")
    model = report["test"]["model"]
    lines.extend([
        "", "## Bootstrap 95% CIs", "",
        f"- AUC: {model['bootstrap_auc_ci_95']}",
        f"- Brier: {model['bootstrap_brier_ci_95']}",
        f"- ECE: {model['bootstrap_ece_ci_95']}",
        f"- False-accept rate: {model['bootstrap_false_accept_ci_95']}",
        f"- Wrong cases by split: {report['wrong_case_count_by_split']}", "",
    ])
    if report.get("warning"):
        lines.append(f"> Warning: {report['warning']}")
    lines.append(f"\nShuffled-label AUC sanity check: {report['sanity_shuffled_label_auc']}")
    lines.append(f"\nShuffled-label near-random: {report['sanity_shuffled_label_near_random']}")
    lines.append(f"\nSynthetic-train / real-test: {report['synthetic_train_real_test']}")
    lines.extend(["", "## Standardized Feature Coefficients", ""])
    for target, pairs in report["feature_importance"].items():
        lines.append(f"### {target}")
        for name, coefficient in pairs[:12]:
            lines.append(f"- `{name}`: {coefficient:+.4f}")
        lines.append("")
    lines.extend(["## Verdict", "", report["verdict"], ""])
    return "\n".join(lines)
