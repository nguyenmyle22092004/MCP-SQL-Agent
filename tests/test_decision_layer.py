import json
from pathlib import Path

import pytest

from app.tracing.validator import validate_answer
from decision_layer.backend import DecisionAnswer
from decision_layer.local.backend import LocalBackend
from decision_layer.local.evaluation import _ece, evaluate_records, load_jsonl
from decision_layer.local.features import FEATURE_NAMES, extract_features
from decision_layer.local.gazetteer import load_gazetteer
from decision_layer.local.model import ARTIFACT_VERSION, LocalModel, _apply_calibrator, _fit_isotonic, _fit_platt, rule_based_probability, train_model
from decision_layer.policy import Action, CostConfig, choose_action, validate_with_decisions
from decision_layer.questions import Choice, Predicate, Score


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "decision_labels.jsonl"
GAZETTEER = {
    "cities": ["Dallas", "Seattle", "Boston", "Atlanta", "San Francisco"],
    "states": ["TX", "WA", "MA", "GA", "CA"],
    "city_state": {"Dallas": "TX", "Seattle": "WA", "Boston": "MA", "Atlanta": "GA", "San Francisco": "CA"},
    "street_names": ["Oak St", "Pine St", "Maple St", "Elm St", "Main St", "Cedar St", "River St", "Hill St", "Sunset St"],
}


def _result(address="100 Oak St", city="Dallas", state="TX", price="300000", bedrooms=3):
    return [{
        "tool": "sql",
        "columns": ["address", "city", "state", "price", "bedrooms"],
        "rows": [[address, city, state, price, bedrooms]],
        "row_count": 1,
        "sql": "SELECT address, city, state, price, bedrooms FROM public.house_prices LIMIT 1",
        "evidence": {"evidence_id": "ev-unit"},
    }]


def _features(answer, *, question="Show this Dallas house.", results=None):
    return extract_features(question, answer, results or _result(), "", "", GAZETTEER)


def test_feature_extraction_covers_swapped_price_and_forbidden_labels():
    results = [{
        "tool": "sql",
        "columns": ["address", "city", "state", "price"],
        "rows": [["100 Oak St", "Dallas", "TX", "300000"], ["200 Pine St", "Seattle", "WA", "350000"]],
        "row_count": 2,
        "evidence": {"evidence_id": "ev-swap"},
    }]
    features = _features(
        "100 Oak St: $350,000; then 200 Pine St: $300,000",
        question="Compare these houses.",
        results=results,
    )

    assert features["n_unmatched_claims"] == 1
    assert features["frac_unmatched"] > 0
    assert len(features) <= 20
    forbidden = {"label", "error_type", "id", "source"}
    assert forbidden.isdisjoint(features)


def test_feature_extraction_detects_wrong_city_and_city_state_pair():
    wrong_city = _features("100 Oak St, Seattle, WA costs $300,000.")
    wrong_state = _features("100 Oak St, Dallas, WA costs $300,000.")

    assert wrong_city["n_cities_not_in_evidence"] == 1
    assert wrong_state["n_states_not_in_evidence"] == 1
    assert wrong_state["city_state_pair_inconsistent"] == 1


def test_address_number_is_not_counted_as_numeric_answer_claim():
    features = _features("100 Oak St, Dallas, TX.")

    assert features["n_addresses_in_answer"] == 1
    assert features["n_numeric_claims"] == 0


def test_feature_extraction_handles_currency_rounding_and_evidence():
    features = _features("100 Oak St, Dallas, TX costs about $787k.", results=_result(price="786554.19"))

    assert features["n_unmatched_claims"] == 0
    assert features["answer_unit_currency_flags"] == 3


def test_expensive_is_vague_only_without_a_metric():
    vague = _features("This is expensive.", question="Find an expensive home.")
    measured = _features("This home is expensive.", question="Find an expensive home by price.")

    assert vague["n_vague_terms"] == 1
    assert measured["n_vague_terms"] == 0


def test_model_training_is_reproducible_and_uses_train_split_only():
    examples = load_jsonl(FIXTURE)
    first = train_model(examples, GAZETTEER, seed=11)
    second = train_model(examples, GAZETTEER, seed=11)

    assert first["training_rows"] == 12
    assert first["training_data_sha256"] == second["training_data_sha256"]
    for target in ("wrong", "needs_clarification"):
        assert first["targets"][target]["coefficients"] == second["targets"][target]["coefficients"]
        assert first["targets"][target]["oof_labels"] == second["targets"][target]["oof_labels"]


def test_training_rejects_rows_without_group_metadata():
    record = load_jsonl(FIXTURE)[0]
    del record["group_id"]

    with pytest.raises(ValueError, match="missing group_id"):
        train_model([record], GAZETTEER)


def test_platt_calibration_improves_ece_on_miscalibrated_toy_scores():
    scores = [0.9] * 10
    labels = [0] * 8 + [1] * 2
    calibrator = _fit_platt(scores, labels)
    calibrated = [_apply_calibrator(calibrator, score) for score in scores]

    assert _ece(labels, calibrated) < _ece(labels, scores)


def test_isotonic_calibration_is_available_and_monotone():
    calibrator = _fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
    calibrated = [_apply_calibrator(calibrator, score) for score in [0.1, 0.2, 0.3, 0.4]]

    assert calibrator["type"] == "isotonic"
    assert calibrated == sorted(calibrated)


def test_model_artifact_load_fails_closed_on_feature_mismatch(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"version": ARTIFACT_VERSION, "feature_names": ["changed"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_names"):
        LocalModel.from_file(artifact_path)


def test_feature_contract_excludes_forbidden_fields():
    assert len(FEATURE_NAMES) <= 20
    assert not ({"label", "error_type", "id", "source"} & set(FEATURE_NAMES))


def test_rule_baselines_cover_wrong_and_clarification_targets():
    wrong_features = {name: 0.0 for name in FEATURE_NAMES}
    wrong_features["n_unmatched_claims"] = 1.0
    vague_features = {name: 0.0 for name in FEATURE_NAMES}
    vague_features["n_vague_terms"] = 1.0

    assert rule_based_probability("wrong", wrong_features) == 1.0
    assert rule_based_probability("needs_clarification", vague_features) == 1.0


def test_action_policy_observes_configured_threshold_and_costs():
    strict = CostConfig(false_accept=10, false_reject=1, clarification=100, wrong_threshold=0.05)
    permissive = CostConfig(false_accept=1, false_reject=10, clarification=100, wrong_threshold=0.8)

    assert choose_action(0.1, 0.0, strict) == Action.REJECT
    assert choose_action(0.1, 0.0, permissive) == Action.ACCEPT


def test_local_backend_missing_artifact_causes_wrapper_skip():
    result = _result()
    question = "Show this Dallas house."
    answer = "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms."
    deterministic = validate_answer(question, answer, result)
    assert deterministic["status"] == "pass"

    decision = validate_with_decisions(
        question,
        answer,
        result,
        "",
        "",
        GAZETTEER,
        LocalBackend("missing-artifact.json"),
    )

    assert decision["action"] == "skipped"
    assert decision["status"] == deterministic["status"]


def test_backend_timeout_is_skipped_and_preserves_deterministic_result():
    class TimeoutBackend:
        def answer(self, question):
            raise TimeoutError("decision backend timeout")

    answer = "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms."
    deterministic = validate_answer("Show this Dallas house.", answer, _result())
    decision = validate_with_decisions(
        "Show this Dallas house.", answer, _result(), "", "", GAZETTEER, TimeoutBackend()
    )

    assert decision["decision_status"] == "skipped"
    assert decision["action"] == Action.SKIPPED.value
    assert decision["status"] == deterministic["status"]
    assert decision["errors"] == deterministic["errors"]


def test_skipped_latency_covers_backend_failure(monkeypatch):
    from decision_layer import policy

    ticks = iter((10.0, 10.75))
    monkeypatch.setattr(policy, "perf_counter", lambda: next(ticks))

    class TimeoutBackend:
        def answer(self, question):
            raise TimeoutError("backend deadline")

    decision = validate_with_decisions(
        "Show this Dallas house.", "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms.",
        _result(), "", "", GAZETTEER, TimeoutBackend(),
    )

    assert decision["decision_status"] == "skipped"
    assert decision["latency_ms"] == 750.0


def test_fake_backend_operates_without_gazetteer_cache_or_database():
    from decision_layer.backend import FakeBackend

    decision = validate_with_decisions(
        "Show this house.",
        "100 Oak St has 3 bedrooms.",
        _result(),
        "",
        "",
        {"cities": [], "states": [], "city_state": {}, "street_names": []},
        FakeBackend(),
    )

    assert decision["decision_status"] == "ok"


def test_policy_falls_back_from_score_to_answer_wrong_predicate():
    class ScoreUnsupportedBackend:
        def answer(self, question):
            if isinstance(question, Score):
                raise NotImplementedError("no score")
            if isinstance(question, Predicate):
                if question.name == "entity_grounding":
                    return DecisionAnswer(value=0.02)
                return DecisionAnswer(value=0.05)
            if isinstance(question, Choice):
                return DecisionAnswer(value="lookup", probabilities={"lookup": 1.0})
            return DecisionAnswer(error="unsupported")

    decision = validate_with_decisions(
        "Show this Dallas house.",
        "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms.",
        _result(),
        "",
        "",
        GAZETTEER,
        ScoreUnsupportedBackend(),
    )

    assert decision["decision_status"] == "ok"
    assert decision["action"] == Action.ACCEPT.value


def test_policy_ask_user_action_is_distinct_from_accept():
    class ClarificationBackend:
        def answer(self, question):
            if isinstance(question, Score):
                return DecisionAnswer(value=0.3)
            if isinstance(question, Predicate):
                return DecisionAnswer(value=question.name == "needs_clarification")
            if isinstance(question, Choice):
                return DecisionAnswer(value="lookup")
            return DecisionAnswer(error="unsupported")

    decision = validate_with_decisions(
        "Find a nice house.", "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms.",
        _result(), "", "", GAZETTEER, ClarificationBackend(),
        CostConfig(clarification=0.0),
    )

    assert decision["action"] == Action.ASK_USER.value


def test_deterministic_failure_can_never_be_accepted():
    class AlwaysCertainBackend:
        def answer(self, question):
            if isinstance(question, Score):
                return DecisionAnswer(value=0.0)
            if isinstance(question, Predicate):
                return DecisionAnswer(value=0.0)
            if isinstance(question, Choice):
                return DecisionAnswer(value="lookup")
            return DecisionAnswer(error="unsupported")

    decision = validate_with_decisions(
        "Show this Dallas house.",
        "100 Oak St, Dallas, TX costs $999,999.",
        _result(),
        "",
        "",
        GAZETTEER,
        AlwaysCertainBackend(),
    )

    assert decision["status"] == "failed"
    assert decision["action"] == Action.REJECT.value


def test_decision_logging_contains_only_summary_and_evidence_hash():
    events = []

    class PassingBackend:
        def answer(self, question):
            if isinstance(question, Score):
                return DecisionAnswer(value=0.01)
            if isinstance(question, Predicate):
                return DecisionAnswer(value=0.02 if question.name == "entity_grounding" else 0.01)
            if isinstance(question, Choice):
                return DecisionAnswer(value="lookup", probabilities={"lookup": 1.0})
            return DecisionAnswer(error="unsupported")

    answer = "100 Oak St, Dallas, TX costs $300,000 with 3 bedrooms."
    result = _result()
    decision = validate_with_decisions(
        "Show this Dallas house.", answer, result, "", "", GAZETTEER, PassingBackend(), logger=events.append
    )

    assert decision["decision_status"] == "ok"
    assert len(events) == 1
    assert "evidence_sha256" in events[0]
    assert "rows" not in events[0]
    assert "100 Oak St" not in json.dumps(events[0])


def test_deterministic_reject_is_logged_without_model_acceptance():
    events = []
    decision = validate_with_decisions(
        "Show this Dallas house.",
        "100 Oak St costs $999,999.",
        _result(),
        "",
        "",
        GAZETTEER,
        LocalBackend("missing.json"),
        logger=events.append,
    )

    assert decision["action"] == Action.REJECT.value
    assert decision["decision_status"] == "deterministic_reject"
    assert events[0]["action"] == Action.REJECT.value


def test_evaluation_fixture_runs_and_writes_markdown(tmp_path):
    records = load_jsonl(FIXTURE)
    report, markdown = evaluate_records(records, GAZETTEER, seed=13, bootstrap_resamples=100)
    output = tmp_path / "evaluation.md"
    output.write_text(markdown, encoding="utf-8")

    assert report["train_rows"] == 12
    assert report["test_rows"] == 8
    assert "## Test Metrics" in output.read_text(encoding="utf-8")
    assert report["by_source"]["synthetic"]["count"] == 8
    assert len(report["train_threshold_sweep"]) == 101
    assert "combined_confusion" in report["by_source"]["synthetic"]
    assert report["wrong_case_count"] < 30


def test_evaluation_rejects_group_leakage_between_train_and_test():
    records = load_jsonl(FIXTURE)
    records[-1]["group_id"] = records[0]["group_id"]

    with pytest.raises(ValueError, match="occurs in both train and test"):
        evaluate_records(records, GAZETTEER, bootstrap_resamples=10)


def test_evaluation_script_cli_writes_markdown_report(tmp_path, monkeypatch):
    from scripts.evaluate_local_model import main

    report_path = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_local_model",
            "--dataset", str(FIXTURE),
            "--gazetteer", str(ROOT / "fixtures" / "decision_gazetteer.json"),
            "--output", str(report_path),
            "--bootstrap-resamples", "25",
        ],
    )

    main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "# Local Decision Model Evaluation" in report_text
    assert "Threshold Sweep" in report_text


def test_default_backend_setting_is_fake():
    from app.config import Settings
    from decision_layer.backend import FakeBackend
    from decision_layer.factory import build_backend

    settings = Settings(_env_file=None)

    assert settings.DECISION_BACKEND == "fake"
    assert isinstance(build_backend(settings), FakeBackend)
    assert settings.DECISION_LAYER_ENABLED is False
    assert settings.DECISION_FAIL_CLOSED is False


def test_local_backend_score_question_raises_clear_error():
    with pytest.raises(NotImplementedError, match="Score"):
        LocalBackend("missing.json").answer(Score("custom_score", {}))


def test_cache_only_gazetteer_load_fails_closed_without_db_access(tmp_path):
    with pytest.raises(FileNotFoundError, match="Gazetteer cache not found"):
        load_gazetteer(tmp_path / "missing.json", build_if_missing=False)
