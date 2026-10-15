from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_layer.local.evaluation import evaluate_records, load_jsonl
from decision_layer.local.gazetteer import load_gazetteer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local decision model on the held-out test split.")
    parser.add_argument("--dataset", default="data/decision_labels.jsonl", help="Labeled JSONL; must contain train and test splits.")
    parser.add_argument("--gazetteer", default="data/gazetteer.json")
    parser.add_argument("--output", default="reports/local_decision_evaluation.md")
    parser.add_argument("--false-accept-cost", type=float, default=10.0)
    parser.add_argument("--false-reject-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise SystemExit(
            f"Labeled dataset not found: {dataset_path}. No labeled JSONL exists in this checkout; "
            "use --dataset tests/fixtures/decision_labels.jsonl only to smoke-test the pipeline."
        )
    try:
        records = load_jsonl(dataset_path)
        gazetteer = load_gazetteer(args.gazetteer)
        report, markdown = evaluate_records(
            records,
            gazetteer,
            false_accept_cost=args.false_accept_cost,
            false_reject_cost=args.false_reject_cost,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except (OSError, ValueError, PermissionError) as exc:
        raise SystemExit(str(exc)) from exc
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nMarkdown report saved to {output_path}")


if __name__ == "__main__":
    main()
